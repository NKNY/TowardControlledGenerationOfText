# TODO Confirm dropout is used everywhere where needed
# TODO Docstrings
# TODO Word embs not trained after pretraining
# TODO Confirm that gradient tape is located where needed
# TODO Implement word2idx
# TODO Implement dataset
# TODO Consider moving embedding of x into an earlier step
# TODO Confirm whether temperature should change during pretraining

from itertools import chain

import tensorflow as tf
import tensorflow_probability as tfp

from utils import temperature, KL_Loss, get_sequential_mask, CrossEntropyWithLogitsLoss

START_TOKEN = "start_token"
EOS_TOKEN = "eos_token"
UNK_TOKEN = "unk_token"
PAD_TOKEN = "pad_token"

class Hu2017(tf.keras.Model):

    def __init__(self, d_emb, d_content, d_style, dropout_rate, discriminator_dropout_rate, num_tokens, word2idx,
                 style_dist_type, style_dist_params, max_timesteps, discriminator_params, optimizer, loss_weights,
                 probs_sum_to_one=True, gradient_norm_clip=5):
        super().__init__()

        self.embedding_layer = EmbeddingLayer(num_tokens, d_emb, word2idx)
        self.encoder = Encoder(d_emb, d_content, dropout_rate)
        self.generator = Generator(d_emb, d_content, dropout_rate, num_tokens,
                                   style_dist_type, style_dist_params,
                                   self.embedding_layer, max_timesteps)
        self.discriminator = Discriminator(d_style, dropout_rate=discriminator_dropout_rate, **discriminator_params)

        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.gradient_norm_clip = gradient_norm_clip

        self.step = 0

        self.loss_obj = {
            'reconstruction': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'KL': KL_Loss(),
            'content': tf.keras.losses.MeanSquaredError(),
            'style': CrossEntropyWithLogitsLoss(probs_sum_to_one)
        }

    def call(self, x, training, mask, sample_style_prior=False, sample_content_prior=False):

        x_emb = self.embedding_layer(x)  # (batch_size, max_timesteps, d_emb)
        batch_size, max_timesteps, d_emb = tf.shape(x_emb)

        # mean.shape, logvar.shape == batch_size, batch_size
        # content.shape == (batch_size, max_timesteps, d_content)
        if sample_content_prior:
            mean, logvar = None, None
            content = self.generator.sample_content_prior(batch_size)
        else:
            mean, logvar = self.encoder(x_emb, mask=mask)
            content = self.generator.sample_content(mean, logvar)

        # style.shape == (batch_size, max_timesteps, d_style)
        if sample_style_prior:
            style = self.generator.sample_style_prior(batch_size)
        else:
            style = self.discriminator(x_emb, training)

        # preds.shape == (batch_size, max_timesteps, num_tokens)
        preds, _ = self.generator.decoder(x_emb, content, style, mask, training)

        return preds, mean, logvar

    def train_step(self, x, targets, pretraining=False):

        if pretraining:
            with tf.GradientTape() as tape:
                pretrain_loss = self.pretrain_step(x)
                gradients = tape.gradient(pretrain_loss, self.get_trainable_variables('pretrain'))
                gradients = [tf.clip_by_norm(g, self.gradient_norm_clip) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables('pretrain')))

            # if self.step % 10 == 0:
            #     print(f'Step {self.step}.\n\tPretrain loss: {pretrain_loss}')
        else:
            self.step = 0

            with tf.GradientTape() as tape:
                discriminator_loss = self.train_discriminator(x, targets)
                gradients = tape.gradient(discriminator_loss, self.get_trainable_variables('discriminator'))
                gradients = [tf.clip_by_norm(g, self.gradient_norm_clip) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables('discriminator')))

            with tf.GradientTape() as tape:
                generator_loss = self.train_generator(x)
                gradients = tape.gradient(generator_loss, self.get_trainable_variables('generator'))
                gradients = [tf.clip_by_norm(g, self.gradient_norm_clip) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables('generator')))

            with tf.GradientTape() as tape:
                encoder_loss = self.train_encoder(x)
                gradients = tape.gradient(encoder_loss, self.get_trainable_variables('encoder'))
                gradients = [tf.clip_by_norm(g, self.gradient_norm_clip) for g in gradients]
                self.optimizer.apply_gradients(zip(gradients, self.get_trainable_variables('encoder')))

            if self.step % 10 == 0:
                print(f'Step {self.step}.\n\tDiscriminator loss: {discriminator_loss}\n\tGenerator loss: {generator_loss}'
                      f'\n\tEncoder_loss: {encoder_loss}')

        self.step += 1

    def generate_autoencoder_targets(self, x):

        batch_size, max_timesteps = tf.shape(x)

        # targets.shape == (batch_size, max_timesteps)
        targets = tf.concat([
            x[:, 1:],
            tf.broadcast_to(self.embedding_layer.word2idx[PAD_TOKEN], (batch_size, 1))
        ], axis=1)

        return targets

    def get_trainable_variables(self, group_name):
        trainable_variable_groups = {
            'pretrain': list(chain(self.encoder.trainable_variables, self.generator.trainable_variables)),
            'discriminator': [x for x in self.discriminator.trainable_variables if not x.name.endswith('embeddings:0')],
            'generator': [x for x in self.generator.trainable_variables if not x.name.endswith('embeddings:0')],
            'encoder': [x for x in self.encoder.trainable_variables if not x.name.endswith('embeddings:0')]
        }
        return trainable_variable_groups[group_name]

    def pretrain_step(self, x):

        targets = self.generate_autoencoder_targets(x)  # (batch_size, max_timesteps)
        mask = get_sequential_mask(x, self.embedding_layer.word2idx[PAD_TOKEN])

        preds, mean, logvar = self(x, training=True, mask=mask, sample_style_prior=True,
                           sample_content_prior=False)  # (batch_size, max_timesteps, num_tokens)

        reconstruction_loss = self.loss_obj['reconstruction'](targets, preds, sample_weight=mask)
        kl_loss = self.loss_obj['KL'](mean, logvar)

        loss = reconstruction_loss + self.loss_weights['KL'] * kl_loss
        # if self.step % 10 == 0:
        #     print(f'Step {self.step}.\n\tReconstruction loss: {reconstruction_loss}\n\tKL loss: {kl_loss}')

        return loss

    def train_encoder(self, x):

        targets = self.generate_autoencoder_targets(x)  # (batch_size, max_timesteps)
        mask = get_sequential_mask(x, self.embedding_layer.word2idx[PAD_TOKEN])

        # (batch_size, max_timesteps, num_tokens), (batch_size, d_content), (batch_size, d_content)
        preds, mean, logvar = self(x, training=True, mask=mask, sample_style_prior=False, sample_content_prior=False)

        reconstruction_loss = self.loss_obj['reconstruction'](targets, preds, sample_weight=mask)
        kl_loss = self.loss_obj['KL'](mean, logvar)

        loss = reconstruction_loss + self.loss_weights['KL'] * kl_loss

        return loss

    def train_generator(self, x):

        batch_size = tf.shape(x)[0]
        temp = temperature(self.step)

        targets = self.generate_autoencoder_targets(x)  # (batch_size, max_timesteps)
        mask = get_sequential_mask(x, self.embedding_layer.word2idx[PAD_TOKEN])

        preds, mean, logvar = self(x, training=True, mask=mask, sample_style_prior=False,
                           sample_content_prior=False)  # (batch_size, max_timesteps, num_tokens)

        reconstruction_loss = self.loss_obj['reconstruction'](targets, preds, sample_weight=mask)
        kl_loss = self.loss_obj['KL'](mean, logvar)

        # (batch_size, max_timesteps, d_emb), (batch_size, d_content), (batch_size, d_content)
        x_sampled, content_sampled, style_sampled = self.generator(soft_embeds=True, batch_size=batch_size, temp=temp)

        # TODO Ensure that looking at mean z vector is correct
        mean_content_sampled, _ = self.encoder(x_sampled, mask)  # (batch_size, d_content)
        preds_style_sampled = self.discriminator(x_sampled, training=True)  # (batch_size, d_style)

        content_loss = self.loss_obj['content'](content_sampled, mean_content_sampled)
        style_loss = self.loss_obj['style'](style_sampled, preds_style_sampled)

        loss = (reconstruction_loss
                + self.loss_weights['KL'] * kl_loss
                + self.loss_weights['style'] * style_loss
                + self.loss_weights['content'] * content_loss)

        return loss

    def train_discriminator(self, x, targets):

        batch_size = tf.shape(x)[0]
        temp = temperature(self.step)

        x_emb = self.embedding_layer(x)  # (batch_size, max_timesteps, d_emb)

        # IMPORTANT: Compared to the PyTorch implementation, style_sampled is not argmax'd
        # (batch_size, max_timesteps, d_emb), (batch_size, d_content), (batch_size, d_style)
        x_sampled, _, style_sampled = self.generator(soft_embeds=False, batch_size=batch_size, temp=temp)
        x_sampled_emb = self.embedding_layer(x_sampled)  # (batch_size, max_timesteps, d_emb)

        preds = self.discriminator(x_emb, True)  # (batch_size, d_style)
        preds_sampled = self.discriminator(x_sampled_emb, True)  # (batch_size, d_style)

        entropy = -tf.reduce_mean(tf.nn.log_softmax(preds_sampled, axis=1))
        loss_s = self.loss_obj['style'](targets, preds)
        loss_u = self.loss_obj['style'](style_sampled, preds_sampled)

        loss = loss_s + self.loss_weights['u']*(loss_u + self.loss_weights['beta']*entropy)

        return loss

    def print_sampled_sentence(self, content=None, style=None):

        temp = temperature(self.step)

        # x_sampled.shape == (1, max_timesteps, d_emb)
        x_sampled = self.generator(soft_embeds=False, content=content, style=style, batch_size=1, temp=temp)[0][0]
        x_sentence = " ".join([self.embedding_layer.idx2word[x.numpy()] for x in x_sampled])

        return x_sentence



class Encoder(tf.keras.layers.Layer):

    def __init__(self, d_emb, d_content, dropout_rate):
        super().__init__()

        self.encoder = tf.keras.layers.LSTM(
            units=d_emb,
            dropout=dropout_rate,
            recurrent_dropout=dropout_rate,
            return_sequences=False,
        )
        self.mean = tf.keras.layers.Dense(d_content)
        self.logvar = tf.keras.layers.Dense(d_content)

    def call(self, x, mask):

        output = self.encoder(x, mask=mask)  # (batch_size, d_emb)
        mean = self.mean(output)  # (batch_size, d_content)
        logvar = self.mean(output)  # (batch_size, d_content)

        return mean, logvar



class Decoder(tf.keras.layers.Layer):

    def __init__(self, d_emb, dropout_rate, num_tokens):
        super().__init__()

        self.num_tokens = num_tokens

        # IMPORTANT: Compared to the PyTorch implementation, no word level dropout is applied.
        # Instead of replacing each dropped out word with an <unk> character, dropout is
        # applied to token embeddings.
        self.decoder = tf.keras.layers.LSTM(
            units=d_emb,
            return_sequences=True,
            return_state=True
        )

        self.embedding_dropout = tf.keras.layers.Dropout(dropout_rate)

        self.scores = tf.keras.layers.Dense(num_tokens)

    def call(self, x, content, style, mask, training, initial_state=None):

        batch_size, max_timesteps, d_emb = tf.shape(x)
        _, d_content = tf.shape(content)
        _, d_style = tf.shape(style)

        # inputs_concat.shape == (batch_size, d_emb+d_content+d_style)
        _x = self.embedding_dropout(x)
        _content = tf.broadcast_to(content[:, tf.newaxis], (batch_size, max_timesteps, d_content))
        _style = tf.broadcast_to(style[:, tf.newaxis], (batch_size, max_timesteps, d_style))
        inputs_concat = tf.concat([_x, _content, _style], axis=-1)

        rnn_outputs = self.decoder(inputs_concat, training=training, initial_state=initial_state, mask=mask)

        # decoder_outputs.shape == (batch_size, seq_len, d_emb)
        # hidden_state[0/1].shape == (batch_size, d_emb)
        decoder_outputs, hidden_state = rnn_outputs[0], rnn_outputs[1:]
        _decoder_outputs = tf.reshape(decoder_outputs, (-1, d_emb))  # (batch_size*max_timesteps, d_emb)

        # scores.shape == (batch_size, max_timesteps,
        _scores = self.scores(_decoder_outputs)  # (batch_size*max_timesteps, num_tokens)
        scores = tf.reshape(_scores, (batch_size, max_timesteps, self.num_tokens))

        return scores, hidden_state



class EmbeddingLayer(tf.keras.layers.Layer):

    def __init__(self, num_tokens, d_emb, word2idx):
        super().__init__()
        self.embeddings = tf.keras.layers.Embedding(num_tokens, d_emb)
        self.word2idx = word2idx
        self.idx2word = {v:k for k,v in word2idx.items()}

    def call(self, x):
        return self.embeddings(x)

class Generator(tf.keras.layers.Layer):

    def __init__(self, d_emb, d_content, dropout_rate, num_tokens, style_dist_type,
                 style_dist_params, embedding_layer, max_timesteps):
        super().__init__()

        self.d_content = d_content
        self.style_dist = self.init_style_dist(style_dist_type, **style_dist_params)
        self.decoder = Decoder(d_emb, dropout_rate, num_tokens)
        self.embedding_layer = embedding_layer
        self.max_timesteps = max_timesteps

    def call(self, soft_embeds, content=None, style=None, *args, **kwargs):
        if soft_embeds:
            return self.generate_soft_embeds(content, style, *args, **kwargs)
        else:
            return self.generate_sentences(content, style, *args, **kwargs)

    @staticmethod
    def init_style_dist(style_dist_type, **style_dist_params):
        return style_dist_type(**style_dist_params)

    def sample_style_prior(self, batch_size):
        return self.style_dist.sample(batch_size)

    def sample_content_prior(self, batch_size):
        content = tf.random.normal(shape=(batch_size, self.d_content))
        return content

    def sample_content(self, mean, logvar):
        eps = tf.random.normal([self.d_content])
        return mean + tf.exp(logvar*0.5) * eps

    def generate_sentences(self, content, style, batch_size, temp, training, squeeze_output=False):
        if content is None:
            content = self.sample_content_prior(batch_size)
        if style is None:
            style = self.sample_style_prior(batch_size)

        output = []

        # last_word.shape == (batch_size, 1)
        last_word = tf.broadcast_to(self.embedding_layer.word2idx[START_TOKEN], (batch_size, 1))
        hidden_state = None

        for i in range(self.max_timesteps):
            last_word_emb = self.embedding_layer(last_word)  # (batch_size, 1, d_emb)

            # scores.shape == (batch_size, 1, num_tokens)
            # hidden_state[0/1] = (batch_size, d_emb)
            scores, hidden_state = self.decoder(last_word_emb, content, style, training, initial_state=hidden_state)
            sampling_dist = tfp.distributions.Categorical(
                logits=scores/temp
            )

            last_word = sampling_dist.sample(1)[0]  # (batch_size, 1)
            output.append(last_word)

        output = tf.concat(output, axis=1)  # (batch_size, max_timesteps)
        if squeeze_output and tf.shape(output)[0] == 1:
            output = tf.squeeze(output, 0)  # (max_timesteps)

        return output, content, style


    def generate_soft_embeds(self, content, style, batch_size, temp, training, squeeze_output=False):

        if content is None:
            content = self.sample_content_prior(batch_size)  # (batch_size, d_content)
        if style is None:
            style = self.sample_style_prior(batch_size)  # (batch_size, d_style)

        output = []

        # last_word.shape == (batch_size, 1)
        last_word = tf.broadcast_to(self.embedding_layer.word2idx[START_TOKEN], (batch_size, 1))
        last_word_emb = self.embedding_layer(last_word)  # (batch_size, 1, d_emb)

        hidden_state = None

        for i in range(self.max_timesteps):

            # scores.shape == (batch_size, 1, num_tokens)
            # hidden_state[0/1] = (batch_size, d_emb)
            scores, hidden_state = self.decoder(last_word_emb, content, style, training, initial_state=hidden_state)
            softmax_scores = tf.nn.softmax(scores/temp)  # (batch_size, 1, num_tokens)
            last_word_emb = tf.matmul(softmax_scores, self.embedding_layer.weights[0])  # (batch_size, 1, d_emb)
            output.append(last_word_emb)

        output = tf.concat(output, axis=1)  # (batch_size, max_timesteps, d_emb)
        return output, content, style

class Discriminator(tf.keras.layers.Layer):

    def __init__(self, d_style, num_kernels, dropout_rate,
                 activation='relu', ngram_sizes=[3,4,5]):
        super().__init__()

        self.convs = [
            tf.keras.layers.Conv1D(
                filters=num_kernels,
                kernel_size=x,
                padding='valid',
                activation=activation
            ) for x in ngram_sizes
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.linear = tf.keras.layers.Dense(d_style)
        self.maxpool = tf.keras.layers.GlobalMaxPool1D()

    def call(self, x, training):

        ngram_outputs = []
        for conv in self.convs:
            conv_out = conv(x)  # (batch_size, max_timesteps-conv.kernel_size+1, num_kernels)
            conv_out_reduced = self.maxpool(conv_out)  # (batch_size, num_kernels)
            ngram_outputs.append(conv_out_reduced)

        ngram_outputs_concat = tf.concat(ngram_outputs, axis=-1)  # (batch_size, num_kernels*len(ngram_sizes))
        scores = self.linear(self.dropout(ngram_outputs_concat, training=training))  # (batch_size, d_style)

        return scores

if __name__ == "__main__":

    import numpy as np

    d_emb = 3
    d_content = 6
    d_style = 2
    num_kernels = 1
    dropout_rate = 0.
    discriminator_dropout_rate = 0.5
    batch_size = 2
    max_timesteps = 7
    num_tokens = 5
    word2idx = {'start_token': 3, 'hello': 1, 'world': 2, 'pad_token': 0, 'eos_token': 4}
    style_dist_type = tfp.distributions.Multinomial
    style_dist_params = {'total_count': num_tokens, 'probs': [0.5, 0.5]}
    discriminator_params = {'num_kernels': num_kernels, 'activation': 'relu', 'ngram_sizes': [3, 4, 5]}
    optimizer = tf.keras.optimizers.Adam()
    loss_weights = {'KL': 1, 'style': 1, 'content': 1, 'u': 1, 'beta': 1}

    m = Hu2017(d_emb, d_content, d_style, dropout_rate, discriminator_dropout_rate, num_tokens, word2idx,
                     style_dist_type, style_dist_params, max_timesteps, discriminator_params, optimizer,
                     loss_weights)

    x = np.random.randint(0, num_tokens, (batch_size, max_timesteps))
    targets = np.random.uniform(0, 1, (batch_size, d_style)).astype(dtype=np.float32)
    mask = x.astype(np.bool).astype(np.float32)

    with tf.GradientTape() as tape:
        pretrain_loss = m.pretrain_step(x)
        gradients = tape.gradient(pretrain_loss, m.get_trainable_variables('pretrain'))
        gradients = [tf.clip_by_norm(g, 5) for g in gradients]
        m.optimizer.apply_gradients(zip(gradients, m.get_trainable_variables('pretrain')))

    with tf.GradientTape() as tape:
        discriminator_loss = m.train_discriminator(x, targets)
        gradients = tape.gradient(discriminator_loss, m.get_trainable_variables('discriminator'))
        gradients = [tf.clip_by_norm(g, 5) for g in gradients]
        m.optimizer.apply_gradients(zip(gradients, m.get_trainable_variables('discriminator')))


    # for i in range(2000):
    #     m.train_step(x, targets, pretraining=True)
    #     if i % 10 == 0:
    #         print(m.print_sampled_sentence())

    # for i in range(100):
    #     m.train_step(x, targets, pretraining=False)
    #     if i % 10 == 0:
    #         print(m.print_sampled_sentence())