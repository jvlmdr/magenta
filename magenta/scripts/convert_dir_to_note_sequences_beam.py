# Copyright 2023 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r""""Converts music files to NoteSequence protos and writes TFRecord file.

Currently supports MIDI (.mid, .midi) and MusicXML (.xml, .mxl) files.

Example usage:
  $ python magenta/scripts/convert_dir_to_note_sequences.py \
    --input_dir=/path/to/input/dir \
    --output_file=/path/to/tfrecord/file \
    --log=INFO
"""

import hashlib
import os

import apache_beam as beam
import note_seq
from note_seq import abc_parser
from note_seq import midi_io
from note_seq import musicxml_reader
import pretty_midi
import tensorflow.compat.v1 as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory containing files to convert.')
tf.app.flags.DEFINE_string('output_file', None,
                           'Path to output TFRecord file. Will be overwritten '
                           'if it already exists.')
tf.app.flags.DEFINE_integer('num_output_shards', 0, 'Number of shards for output file.')
tf.app.flags.DEFINE_bool('recursive', False,
                         'Whether or not to recurse into subdirectories.')
tf.app.flags.DEFINE_string('log', 'INFO',
                           'The threshold for what messages will be logged '
                           'DEBUG, INFO, WARN, ERROR, or FATAL.')
tf.app.flags.DEFINE_float('max_tick', 1e7,
                          'Modify the value of pretty_midi.MAX_TICK.')


def generate_note_sequence_id(filename, collection_name, source_type):
  """Generates a unique ID for a sequence.

  The format is:'/id/<type>/<collection name>/<hash>'.

  Args:
    filename: The string path to the source file relative to the root of the
        collection.
    collection_name: The collection from which the file comes.
    source_type: The source type as a string (e.g. "midi" or "abc").

  Returns:
    The generated sequence ID as a string.
  """
  filename_fingerprint = hashlib.sha1(filename.encode('utf-8'))
  return '/id/%s/%s/%s' % (
      source_type.lower(), collection_name, filename_fingerprint.hexdigest())


def iter_files(root_dir, sub_dir, recursive=False):
  """Yields file paths relative to root_dir."""
  dir_to_convert = os.path.join(root_dir, sub_dir)
  files_in_dir = tf.gfile.ListDirectory(os.path.join(dir_to_convert))
  recurse_sub_dirs = []
  for file_in_dir in files_in_dir:
    _, suffix = os.path.splitext(file_in_dir)
    if suffix.lower() in ['.mid', '.midi', '.xml', '.mxl', '.abc']:
      yield os.path.join(sub_dir, file_in_dir)
      continue

    full_file_path = os.path.join(root_dir, sub_dir, file_in_dir)
    if recursive and tf.gfile.IsDirectory(full_file_path):
      recurse_sub_dirs.append(os.path.join(sub_dir, file_in_dir))
      continue

    tf.logging.warning('Unable to find a converter for file %s', full_file_path)

  for recurse_sub_dir in recurse_sub_dirs:
    yield from iter_files(root_dir, recurse_sub_dir, recursive)


class ConvertFile(beam.DoFn):

  def __init__(self, root_dir):
    self.root_dir = root_dir

  def process(self, rel_file_path):
    return convert_file(self.root_dir, rel_file_path)


def convert_file(root_dir, rel_file_path):
  """Converts file to one or multiple NoteSequences.

  Args:
    root_dir: A string specifying a root directory.
    sub_dir: A string specifying a path to a directory under `root_dir` in which
        to convert contents.

  Returns:
    A list of NoteSequences for the file.
  """
  full_file_path = os.path.join(root_dir, rel_file_path)
  if (rel_file_path.lower().endswith('.mid') or
      rel_file_path.lower().endswith('.midi')):
    try:
      sequence = convert_midi(root_dir, rel_file_path)
    except Exception as exc:  # pylint: disable=broad-except
      tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
      return []
    return [sequence] if sequence else []

  if (rel_file_path.lower().endswith('.xml') or
      rel_file_path.lower().endswith('.mxl')):
    try:
      sequence = convert_musicxml(root_dir, rel_file_path)
    except Exception as exc:  # pylint: disable=broad-except
      tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
      return []
    return [sequence] if sequence else []

  if rel_file_path.lower().endswith('.abc'):
    try:
      sequences = convert_abc(root_dir, rel_file_path)
    except Exception as exc:  # pylint: disable=broad-except
      tf.logging.fatal('%r generated an exception: %s', full_file_path, exc)
      return []
    return sequences

  tf.logging.fatal('unknown file type: %s', full_file_path)
  return []


def convert_midi(root_dir, rel_file_path):
  """Converts a midi file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  full_file_path = os.path.join(root_dir, rel_file_path)
  try:
    sequence = midi_io.midi_to_sequence_proto(
        tf.gfile.GFile(full_file_path, 'rb').read())
  except midi_io.MIDIConversionError as e:
    tf.logging.warning(
        'Could not parse MIDI file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None
  sequence.collection_name = os.path.basename(root_dir)
  # sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.filename = rel_file_path
  sequence.id = generate_note_sequence_id(
      sequence.filename, sequence.collection_name, 'midi')
  tf.logging.info('Converted MIDI file %s.', full_file_path)
  return sequence


def convert_musicxml(root_dir, rel_file_path):
  """Converts a musicxml file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  full_file_path = os.path.join(root_dir, rel_file_path)
  try:
    sequence = musicxml_reader.musicxml_file_to_sequence_proto(full_file_path)
  except musicxml_reader.MusicXMLConversionError as e:
    tf.logging.warning(
        'Could not parse MusicXML file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None
  sequence.collection_name = os.path.basename(root_dir)
  # sequence.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
  sequence.filename = rel_file_path
  sequence.id = generate_note_sequence_id(
      sequence.filename, sequence.collection_name, 'musicxml')
  tf.logging.info('Converted MusicXML file %s.', full_file_path)
  return sequence


def convert_abc(root_dir, rel_file_path):
  """Converts an abc file to a sequence proto.

  Args:
    root_dir: A string specifying the root directory for the files being
        converted.
    full_file_path: the full path to the file to convert.

  Returns:
    Either a NoteSequence proto or None if the file could not be converted.
  """
  full_file_path = os.path.join(root_dir, rel_file_path)
  try:
    tunes, exceptions = abc_parser.parse_abc_tunebook(
        tf.gfile.GFile(full_file_path, 'rb').read())
  except abc_parser.ABCParseError as e:
    tf.logging.warning(
        'Could not parse ABC file %s. It will be skipped. Error was: %s',
        full_file_path, e)
    return None

  for exception in exceptions:
    tf.logging.warning(
        'Could not parse tune in ABC file %s. It will be skipped. Error was: '
        '%s', full_file_path, exception)

  sequences = []
  for idx, tune in tunes.items():
    tune.collection_name = os.path.basename(root_dir)
    # tune.filename = os.path.join(sub_dir, os.path.basename(full_file_path))
    tune.filename = rel_file_path
    tune.id = generate_note_sequence_id(
        '{}_{}'.format(tune.filename, idx), tune.collection_name, 'abc')
    sequences.append(tune)
    tf.logging.info('Converted ABC file %s.', full_file_path)
  return sequences


def main(argv):
  tf.logging.set_verbosity(FLAGS.log)
  pretty_midi.pretty_midi.MAX_TICK = FLAGS.max_tick

  pipeline_args = argv[1:]
  tf.logging.info("pipeline_args: %s", pipeline_args)

  if not FLAGS.input_dir:
    tf.logging.fatal('--input_dir required')
    return
  if not FLAGS.output_file:
    tf.logging.fatal('--output_file required')
    return

  input_dir = os.path.expanduser(FLAGS.input_dir)
  output_file = os.path.expanduser(FLAGS.output_file)
  output_dir = os.path.dirname(output_file)

  if output_dir:
    tf.gfile.MakeDirs(output_dir)

  pipeline_options = beam.options.pipeline_options.PipelineOptions(pipeline_args)

  input_files = list(iter_files(input_dir, '', FLAGS.recursive))
  tf.logging.info('found %d files', len(input_files))

  with beam.Pipeline(options=pipeline_options) as p:
    p |= 'file_list' >> beam.Create(input_files)
    p |= 'shuffle_input' >> beam.Reshuffle()
    p |= 'convert_file' >> beam.ParDo(ConvertFile(input_dir))
    p |= 'shuffle_output' >> beam.Reshuffle()
    p |= 'write' >> beam.io.WriteToTFRecord(
        FLAGS.output_file, num_shards=FLAGS.num_output_shards,
        coder=beam.coders.ProtoCoder(note_seq.NoteSequence))


def console_entry_point():
  tf.disable_v2_behavior()
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
