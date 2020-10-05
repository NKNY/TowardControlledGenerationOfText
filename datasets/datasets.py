import os

import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

START_TOKEN = "SOS"
EOS_TOKEN = "EOS"
UNK_TOKEN = "UNK"
PAD_TOKEN = ""

class SST:
    def __init__(self, max_timesteps, shuffle_buffer_size, dataset_version='hu', dataset_dir=None, verbose=True):

        self.max_timesteps = max_timesteps
        self.shuffle_buffer_size = shuffle_buffer_size
        self.d_c = 2

        self.splits = {}

        self.dataset = self.load_dataset(dataset_version, dataset_dir)
        self.init_tokenizer(self.dataset['train'], verbose=verbose)
        self.encoder = tfds.features.text.TokenTextEncoder(self.vocabulary, oov_token=UNK_TOKEN)
        self.init_token_idx()

    def init_token_idx(self):
        token2idx = {x: self.encoder.encode(x)[0] for x in self.vocabulary}
        token2idx[UNK_TOKEN] = self.num_tokens - 1
        token2idx[PAD_TOKEN] = 0
        self.token_idx = {
            'token2idx': token2idx,
            'idx2token': {v: k for k, v in token2idx.items()}
        }

    def load_dataset(self, dataset_version, dataset_dir=None):

        # Create a description of the features.
        feature_description = {
            'sentence': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'label': tf.io.FixedLenFeature([], tf.int64, default_value=0)
        }

        def _parse_function(example_proto):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, feature_description)

        assert dataset_version in ['glue', 'hu']

        if dataset_version == 'glue':
            return tfds.load('glue/sst2')

        elif dataset_version == 'hu':
            subsets = ['train', 'validation', 'test']
            assert dataset_dir is not None and os.path.exists(dataset_dir) and \
                   all([f'{x}.tfrecords' in os.listdir(dataset_dir) for x in subsets])

            subset_paths = {s: os.path.join(dataset_dir, f'{s}.tfrecords') for s in subsets}
            raw_subsets = {s: tf.data.TFRecordDataset([p]) for s, p in subset_paths.items()}
            dataset = {s: d.map(_parse_function) for s,d in raw_subsets.items()}

            return dataset

    def __call__(self, split, batch_size):

        padded_shapes = ([self.max_timesteps], [self.d_c])
        padding_values = (tf.cast(self.token_idx['token2idx'][PAD_TOKEN], dtype=tf.int64), None)

        return self.dataset[split]\
            .map(self.encode_input_map_fn)\
            .filter(self.filter_on_len_lambda(self.max_timesteps))\
            .map(self.encode_label)\
            .shuffle(self.shuffle_buffer_size) \
            .repeat()\
            .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)


    def init_tokenizer(self, corpus, verbose=True):

        self.tokenizer = tfds.features.text.Tokenizer()

        vocabulary = set()
        corpus = tqdm(corpus) if verbose else corpus

        for sample in corpus:
            sample_tokens = self.tokenizer.tokenize(sample['sentence'].numpy())
            if len(sample_tokens) <= self.max_timesteps - 2:
                vocabulary.update(sample_tokens)

        vocabulary = [START_TOKEN, EOS_TOKEN] + sorted(list(vocabulary))

        self.vocabulary = vocabulary
        self.num_tokens = len(vocabulary) + 2  # +1 for PAD, +1 for UNK


    def encode_input(self, text_tensor, label):
        encoded_text = self.encoder.encode(str.encode(START_TOKEN + " ") + text_tensor.numpy() + str.encode(" " + EOS_TOKEN))
        return encoded_text, label

    def encode_input_map_fn(self, input):

        text, label = input['sentence'], input['label']

        # py_func doesn't set the shape of the returned tensors.
        encoded_text, label = tf.py_function(self.encode_input,
                                             inp=[text, label],
                                             Tout=(tf.int64, tf.int64))
        # `tf.data.Datasets` work best if all components have a shape set
        #  so set the shapes manually:
        encoded_text.set_shape([None])
        label.set_shape([])

        return encoded_text, label

    def filter_on_len(self, x, y, max_timesteps):
        return tf.shape(x)[0] <= max_timesteps - 2

    def filter_on_len_lambda(self, max_timesteps):
        return lambda x, y: self.filter_on_len(x, y, max_timesteps)

    def encode_label(self, text_tensor, label):
        label_tensor = tf.one_hot(label, 2)
        return text_tensor, label_tensor

if __name__ == "__main__":
    ds = SST(100, 20)
    for i in ds('train', 2).take(2):
        print(i)
    for i in ds('validation', 2).take(2):
        print(i)