import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

START_TOKEN = "SOS"
EOS_TOKEN = "EOS"
UNK_TOKEN = "UNK"
PAD_TOKEN = ""

class SST:
    def __init__(self, max_timesteps, shuffle_buffer_size, verbose=True):

        self.max_timesteps = max_timesteps
        self.shuffle_buffer_size = shuffle_buffer_size
        self.d_c = 2

        self.splits = {}

        self.dataset = tfds.load('glue/sst2')
        self.init_tokenizer(self.dataset['train'], verbose=verbose)
        self.encoder = tfds.features.text.TokenTextEncoder(self.vocabulary, oov_token=UNK_TOKEN)
        self.init_token_idx()

    def init_token_idx(self):
        token2idx = {x: self.encoder.encode(x)[0] for x in self.vocabulary}
        token2idx[UNK_TOKEN] = 0
        token2idx[PAD_TOKEN] = self.num_tokens - 1
        self.token_idx = {
            'token2idx': token2idx,
            'idx2token': {v: k for k, v in token2idx.items()}
        }

    def __call__(self, split, batch_size, max_unpadded_timesteps):

        padded_shapes = ([self.max_timesteps], [self.d_c])
        padding_values = (tf.cast(self.token_idx['token2idx'][PAD_TOKEN], dtype=tf.int64), None)

        return self.dataset[split]\
            .map(self.encode_input_map_fn)\
            .map(self.encode_label)\
            .shuffle(self.shuffle_buffer_size) \
            .filter(self.filter_on_len_lambda(max_unpadded_timesteps))\
            .repeat()\
            .padded_batch(batch_size, padded_shapes=padded_shapes, padding_values=padding_values)


    def init_tokenizer(self, corpus, verbose=True):

        self.tokenizer = tfds.features.text.Tokenizer()

        vocabulary = set()
        corpus = tqdm(corpus) if verbose else corpus

        for sample in corpus:
            sample_tokens = self.tokenizer.tokenize(sample['sentence'].numpy())
            vocabulary.update(sample_tokens)

        vocabulary = [START_TOKEN, EOS_TOKEN] + list(vocabulary)

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
    for i in ds('train', 2, 15).take(2):
        print(i)
    for i in ds('validation', 2, 15).take(2):
        print(i)