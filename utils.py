import numpy as np
from os import path, makedirs
from datetime import datetime
import pickle
import json
import torch
from torch.optim import Adam, AdamW
from deep_learning.deep_utils import FocalLoss
import sys
import argparse
import subprocess


def train_test_split(n_samples, train_val_test_ratio, random_state: np.random.RandomState = None):
    """
    generates a dataset split.
    :param n_samples: int. number of samples in the dataset
    :param train_val_test_ratio: list. A list of 2 or 3 elements. each stands for the fraction of samples for each set.
    :param random_state: np.random.RandomState. A rondom state for drawing samples.
    :return: 3 numpy arrays
    """
    if random_state:
        split = random_state.choice([0, 1, 2], n_samples, replace=True, p=train_val_test_ratio)
    else:
        split = np.random.choice([0, 1, 2], n_samples, replace=True, p=train_val_test_ratio)
    train_indexes = np.nonzero(split == 0)[0]
    val_indexes = np.nonzero(split == 1)[0]
    test_indexes = np.nonzero(split == 2)[0]
    return train_indexes, val_indexes, test_indexes


def get_pulling_func(pulling_func_name):
    """
    :param pulling_func_name: string. name of a valid pulling function
    :return: model puling function
    """
    from deep_learning.deep_utils import MaskedSum, MaskedMean

    if pulling_func_name == 'mean':
        return MaskedMean
    elif pulling_func_name == 'sum':
        return MaskedSum
    else:
        assert 0, '{} is not a vallid pulling operation function name'.format(pulling_func_name)


def get_loss_function(loss_name, **kwargs):
    """
    :param loss_name: string. A valid loss function name
    :param kwargs: any additional loss function parameters
    :return: method. loss function
    """
    if loss_name == 'BCE':
        return torch.nn.BCEWithLogitsLoss(reduction='sum')
    elif loss_name == 'FOCAL':
        return FocalLoss(gamma=kwargs['focal_gamma'])
    else:
        assert 0, '{} is not a valid loss name'.format(loss_name)


def get_optimizer(optimizer_name):
    """
    :param optimizer_name: string. valid optimizer
    :return: pytorch optimizer
    """
    if optimizer_name == 'ADAMW':
        return AdamW
    elif optimizer_name == 'ADAM':
        return Adam
    else:
        assert 0, '{} is not a valid optimizer'.format(optimizer_name)


def get_root_path():
    """
    :return: string. absolute path of the root folder of the project
    """
    return path.dirname(path.realpath(__file__))


def get_time():
    """
    :return: string. current time. format: day(2)_month(2)_year(4)__hour(2)_minutes(2)_seconds(2)_miliseconds(3)
    """
    return datetime.today().strftime('%d_%m_%Y__%H_%M_%S_%f')[:-3]


def save_propagation_score(propagation_scores, normalization_constants,
                           row_id_to_idx, col_id_to_idx, propagation_args, data_args, filename):
    """

    :param propagation_scores: numpy.ndarray. dimenstions: [#source genes, #effect genes]
    :param normalization_constants: dict. per chosen normalization method
    :param row_id_to_idx: dict. maps genes from row id to gene idx
    :param col_id_to_idx: maps genes from col id to gene idx
    :param propagation_args: args['propagation'] from configuration
    :param data_args: args['data']
    :param filename: filename to save
    :return: string, path where file was saved
    """
    save_path = path.join(get_root_path(), 'input', 'propagation_scores', filename)
    save_dict = {'propagation_args': propagation_args, 'row_id_to_idx': row_id_to_idx, 'col_id_to_idx': col_id_to_idx,
                 'propagation_scores': propagation_scores, 'normalization_constants': normalization_constants,
                 'data_args': data_args, 'data': get_time()}
    save_pickle(save_dict, save_path)
    return save_path


def save_pickle(obj, save_dir):
    with open(save_dir, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(load_dir):
    with open(load_dir, 'rb') as f:
        obj = pickle.load(f)
    return obj


def save_model(path, model):
    torch.save(model.state_dict(), path)


def load_model(path, args, device='cpu'):
    """
    helper function to load deep model
    :param path: string. path of saved model
    :param args: dict. args dict of the run where model was trained.
    :param device: string. current run device
    :return: DeepClassifier object.
    """
    from deep_learning.models import EncoderBlock, DeepClassifier
    state_dict = torch.load(path, map_location=device)
    n_experiments = state_dict['classifier.0.weight'].shape[1]
    deep_prop_model = EncoderBlock(args['model'], n_experiments)
    model = DeepClassifier(deep_prop_model)
    model.load_state_dict(state_dict)
    return model


def redirect_output(log_file_path):
    """
    direct std out and stderr also to log file.
    :param log_file_path: string. path to log file.
    :return:
    """
    sys.stdout = Logger(log_file_path, sys.stdout)
    sys.stderr = Logger(log_file_path, sys.stderr)


class Logger(object):
    """
    A helper class to output std to a file.
    """
    def __init__(self, path, out_type):
        self.terminal = out_type
        self.log = open(path, "a", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass


def str2bool(v):
    """
    helper method to handle boolean arguments
    :param v: string.
    :return: boolean.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def log_project_version(output_path):
    """
    A method for logging project's version, submodules versions and differences from last commit
    :param output_path: string. path to save log.
    :return:
    """
    git_log_path = path.join(output_path, 'git_log')
    makedirs(git_log_path, exist_ok=True)
    version = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    modified_files = subprocess.check_output(['git', 'diff', '--name-only'], text=True).strip().split('\n')

    version_dict = {'commit': version,
                    'modified_files': modified_files}

    if path.isfile('.gitmodules'):
        submodules_dict = {}
        for submodule_entry in subprocess.check_output(['git', 'config', '--file', '.gitmodules',
                                                        '--get-regexp', 'path'], text=True).split('\n'):
            if len(submodule_entry):
                submodule_name = submodule_entry.split(' ')[0].split('.')[1]
                submodule_path = submodule_entry.split(' ')[1]
                submodule_version = subprocess.check_output(['git', '-C', '{}'.format(submodule_path), 'rev-parse', '--short', 'HEAD'], text=True).strip()
                submodules_dict[submodule_name] = submodule_version
        version_dict['submodules'] = submodules_dict

    diff_output = subprocess.check_output(['git', 'diff', '--minimal'], text=True).strip()

    with open(path.join(git_log_path, 'git_version.json'), 'w') as f:
        json.dump(version_dict, f, indent=4, separators=(',', ': '))

    with open(path.join(git_log_path, 'git_diff'), 'w') as f:
        f.writelines(diff_output)

