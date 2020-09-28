import os

from argparse import ArgumentParser
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm

from datasets import SST
from model import Hu2017

def train_Hu2017_SST(dataset_args, model_args, training_loop_args):
    dataset = SST(**dataset_args['init'])
    model = Hu2017(token_idx=dataset.token_idx, **model_args['init'])
    training_loop = TrainingLoop(model, dataset, **training_loop_args['init'])
    training_loop(**training_loop_args['call'])

class Hu2017ArgumentParser:
    def __init__(self):
        self.parser = ArgumentParser()
        self.defaults = {
            'd_emb': {'default': 300, 'type': int},
            'd_content': {'default': 300, 'type': int},
            'd_style': {'default': 2, 'type': int},
            'dropout_rate': {'default': 0., 'type': float},
            'discriminator_dropout_rate': {'default': .5, 'type': float},
            'batch_size': {'default': 15, 'type': int},
            'max_timesteps': {'default': 15, 'type': int},
            'max_unpadded_timesteps': {'default': 15, 'type': int},
            'style_dist_type': {'default': tfp.distributions.Multinomial},
            'style_dist_params': {'default': {'total_count': 1, 'probs': [0.5, 0.5]}},
            'discriminator_params': {'default': {'num_kernels': 100, 'activation': 'relu', 'ngram_sizes': [3, 4, 5]}},
            'optimizer': {'default': tf.keras.optimizers.Adam()},
            'loss_weights': {'default': {'KL': .1, 'style': .1, 'content': .1, 'u': .1, 'beta': .1}},
            'shuffle_buffer_size': {'default': 100, 'type': int},
            'model_dir': {'default': '/Users/nknyazev/Documents/CS/projects/text_style_transfer/models/Hu2017'},
            'num_pretraining_steps': {'default': 10, 'type': int},
            'num_training_steps': {'default': 20, 'type': int},
            'checkpoint_frequency_steps': {'default': 10, 'type': int},
            'log_frequency_steps': {'default': 10, 'type': int},
            'log_dir': {'default': '/Users/nknyazev/Documents/CS/projects/text_style_transfer/models/Hu2017'},
        }
        self.param_groups = {
            'model_args': {
                'init': [
                    'd_emb', 'd_content', 'd_style', 'dropout_rate', 'discriminator_dropout_rate',
                    'style_dist_type', 'style_dist_params', 'max_timesteps', 'discriminator_params', 'optimizer',
                    'loss_weights', 'log_dir', 'log_frequency_steps'
            ],
        },
            'dataset_args': {
                'init': [
                    'max_timesteps', 'shuffle_buffer_size'
                ]
            },
            'training_loop_args': {
                'init': [
                    'model_dir'
                ],
                'call': [
                    'num_training_steps', 'num_pretraining_steps', 'batch_size', 'max_unpadded_timesteps',
                    'checkpoint_frequency_steps', 'log_frequency_steps'
                ]
            }
        }

        for arg, params in self.defaults.items():
            self.parser.add_argument('--' + arg, **params)
        self.args = self.parser.parse_args()
        self.group_params()

    def group_params(self):
        self.param_dict = {
            param_group_key : {
                param_type_key: {attr_name: getattr(self.args, attr_name) for attr_name in param_type_value}
                for param_type_key, param_type_value in param_group_value.items()
            } for param_group_key, param_group_value in self.param_groups.items()
        }

class TrainingLoop:
    def __init__(self, model, dataset, model_dir):
        self.model = model
        self.dataset = dataset
        self.model_dir = model_dir
        self.log_dir = os.path.join(model_dir, 'logs')
        self.checkpoint_prefix = os.path.join(model_dir, 'ckpt')
        self.checkpoint = tf.train.Checkpoint(model=self.model)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, model_dir, 3)

        self.restore_from_checkpoint = os.path.exists(self.checkpoint_prefix)
        if not self.restore_from_checkpoint:
            os.makedirs(self.checkpoint_prefix)
        self.checkpoint_manager.restore_or_initialize()

    def __call__(self, num_pretraining_steps, num_training_steps, batch_size, max_unpadded_timesteps,
                 checkpoint_frequency_steps, log_frequency_steps):

        if self.model.step == 0:
            pretraining_iterator = tqdm(enumerate(self.dataset('train', batch_size, max_unpadded_timesteps).take(num_pretraining_steps)))
            print(f'Starting pretraining for {num_pretraining_steps} steps.')
            for i, (input, targets) in pretraining_iterator:
                losses = self.model.train_vae_step(input)
                if i % log_frequency_steps == 0:
                    pretraining_iterator.set_description(', '.join(f'{k}: {v}' for k, v in losses.items()))
                    print(f'Sampled sentence: {self.model.print_sampled_sentence()}')
                if i % checkpoint_frequency_steps == 0 and i:
                    print(f'Saving checkpoint for pretraining step {i} (global pretrain step: {self.model.pretrain_step - 1}).')
                    self.checkpoint_manager.save()
            if num_pretraining_steps > 0:
                print(f'Pretraining complete. Saving checkpoint to {self.model_dir}.')
                self.checkpoint_manager.save()
        else:
            print('Skipping pretraining.')

        training_iterator = tqdm(enumerate(self.dataset('train', batch_size, max_unpadded_timesteps).take(num_training_steps)))
        print(f'Starting training for {num_training_steps} steps.')
        for i, (input, targets) in training_iterator:
            losses = self.model.train_step(input, targets)
            if i % log_frequency_steps == 0:
                training_iterator.set_description(', '.join(f'{k}: {v}' for k, v in losses.items()))
                print(f'Sampled sentence: {self.model.print_sampled_sentence()}')
            if i % checkpoint_frequency_steps == 0 and i:
                print(f'Saving checkpoint for training step {i} (global step: {self.model.step - 1}).')
                self.checkpoint_manager.save()

        print(f'Training complete. Saving checkpoint to {self.model_dir}.')
        self.checkpoint_manager.save()

if __name__ == '__main__':
    parser = Hu2017ArgumentParser()
    parsed_params = parser.param_dict
    train_Hu2017_SST(**parsed_params)

