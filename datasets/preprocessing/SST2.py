"""
Convert torchtext SST-2 dataset into train, validation and test tfrecords.
"""

import os
import sys

import tensorflow as tf
from torchtext import data, datasets
from tqdm import tqdm

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_example(feature0, feature1):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
      'sentence': _bytes_feature(feature0),
      'label': _int64_feature(feature1),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def convert_label(label):
    return int(label == 'positive')

output_folder = sys.argv[1]

# Keep text and labels as they are without converting into a list
text_field = data.Field(sequential=False)
label_field = data.Field(sequential=False)

filter_pred = lambda example: example.label != 'neutral'  # Keep only positive and negative examples
split_names = ['train', 'validation', 'test']
splits = datasets.SST.splits(text_field, label_field, filter_pred=filter_pred)

for split_name, split in zip(split_names, splits):
    output_path = os.path.join(output_folder, f'{split_name}.tfrecords')
    with tf.io.TFRecordWriter(output_path) as writer:
        for torch_example in tqdm(split):
            text, label = torch_example.text, torch_example.label
            label = convert_label(label)  # Binarise the label
            tf_example = serialize_example(text.encode('utf-8'), label)  # Text serialised as bytes, not str
            writer.write(tf_example)
