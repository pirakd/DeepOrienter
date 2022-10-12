from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.dirname(path.realpath(__file__)))))
from os import makedirs
from deep_learning.data_loaders import LightDataset
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from utils import read_data, load_model, log_results, get_time, get_root_path, train_test_split,\
    gen_propagation_scores, get_loss_function
import torch
import numpy as np
import json
import argparse

def run(sys_args):
    root_path = get_root_path()
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    model_args_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'args')
    model_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'model')
    model_results_path = path.join(root_path, 'input', 'models', sys_args.model_name, 'results')

    with open(model_args_path, 'r') as f:
        args = json.load(f)
    with open(model_results_path, 'r') as f:
        normalization_constants_dicts = json.load(f)['normalization_constants_dicts']

    device = torch.device("cuda:{}".format(sys_args.device) if torch.cuda.is_available() else "cpu")
    train_dataset = args['data']['directed_interactions_filename']
    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['n_experiments'] = sys_args.n_experiments
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    model = load_model(model_path, args).to(device)
    rng = np.random.RandomState(args['data']['random_seed'])

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)

    n_experiments = len(sources.keys())
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, _ = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list, calc_normalization_constants=False)

    train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'], len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                                random_state=rng, directed_interactions=directed_interactions_pairs_list)
    test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                directed_interactions_pairs_list[test_indexes], sources,
                                terminals, args['data']['normalization_method'],
                                samples_normalization_constants=normalization_constants_dicts['samples'],
                                degree_feature_normalization_constants= normalization_constants_dicts['degrees'],
                                pairs_source_type=directed_interactions_source_type[test_indexes],
                                id_to_degree=id_to_degree, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False, )
    intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                               focal_gamma=args['train']['focal_gamma'])
    trainer = ClassifierTrainer(args['train']['n_epochs'], criteria=nn.CrossEntropyLoss(), intermediate_criteria=intermediate_loss_type,
                                intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                optimizer=None, eval_metric=None, eval_interval=args['train']['eval_interval'], device='cpu')


    results = trainer.eval_by_source(model, test_loader) # use trainer.predict to receive predicted interactions

    print('Overall results: {}'.format(results['overall']))

    results_dict = {'test_stats': results, 'n_experiments': n_experiments, 'train_dataset':train_dataset,
                    'test_dataset':args['data']['directed_interactions_filename'], 'model_path': model_path}
    log_results(output_file_path,  args, results_dict)


if __name__ == '__main__':
    model_name = '12_10_2022__15_58_06_294'
    load_prop = True
    save_prop = False
    n_exp = 1
    split = [0, 0, 1]
    interaction_type = 'KPI'
    device = 'cpu'
    prop_scores_filename = 'AML_KPI_1'

    parser = argparse.ArgumentParser()
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-m,', '--model_name', type=str, help='name of saved model folder in input/models', default=model_name)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', type=str, help='KPI/STKE',
                        default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    args = parser.parse_args()
    args.load_prop_scores = True

    run(args)
