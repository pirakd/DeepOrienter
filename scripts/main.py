from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import makedirs
from deep_learning.models import DeepClassifier, EncoderBlock
from deep_learning.data_loaders import LightDataset
import json
from deep_learning.trainer import ClassifierTrainer
from torch import nn
from torch.utils.data import DataLoader
from utils import read_data, get_time, get_root_path, train_test_split, get_loss_function,\
    gen_propagation_scores, redirect_output, get_optimizer, save_model
import torch
import numpy as np
from presets import example_preset
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import argparse

def run(sys_args):
    root_path = get_root_path()
    output_folder = 'output'
    output_file_path = path.join(root_path, output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))
    n_experiments = sys_args.n_experiments
    args = example_preset

    if sys_args.device:
        device = torch.device("cuda:{}".format(sys_args.device))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments']  = n_experiments
    rng = np.random.RandomState(args['data']['random_seed'])
    print(json.dumps(args, indent=4))

    # data read
    network, directed_interactions, sources, terminals, id_to_degree =\
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)

    n_experiments = len(sources.keys())
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    # generating datasets
    train_indexes, val_indexes, test_indexes = train_test_split(args['data']['split_type'], len(directed_interactions_pairs_list), args['train']['train_val_test_split'],
                                                                random_state=rng, directed_interactions=directed_interactions_pairs_list)
    train_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores, directed_interactions_pairs_list[train_indexes],
                                 sources, terminals, args['data']['normalization_method'],
                                 normalization_constants_dict, degree_feature_normalization_constants=None,
                                 pairs_source_type=directed_interactions_source_type, id_to_degree=id_to_degree)
    train_loader = DataLoader(train_dataset, batch_size=args['train']['train_batch_size'], shuffle=True, pin_memory=False, num_workers=sys_args.n_workers)

    degree_normalization_constants = {'lmbda': train_dataset.degree_normalizer.lmbda,
                                      'mean':train_dataset.degree_normalizer.lmbda,
                                      'std':train_dataset.degree_normalizer.std}
    val_dataset = LightDataset(row_id_to_idx, col_id_to_idx,
                               propagation_scores, directed_interactions_pairs_list[val_indexes],
                               sources, terminals, args['data']['normalization_method'],
                               normalization_constants_dict, degree_normalization_constants,
                               directed_interactions_source_type[val_indexes], id_to_degree)
    val_loader = DataLoader(val_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False, num_workers=sys_args.n_workers)
    test_dataset = LightDataset(row_id_to_idx, col_id_to_idx, propagation_scores,
                                directed_interactions_pairs_list[test_indexes],
                                sources, terminals, args['data']['normalization_method'],
                                normalization_constants_dict, degree_normalization_constants,
                                directed_interactions_source_type[test_indexes], id_to_degree)
    test_loader = DataLoader(test_dataset, batch_size=args['train']['test_batch_size'], shuffle=False, pin_memory=False, num_workers=sys_args.n_workers)

    models_list = []
    train_stats_list = []
    for i in range(sys_args.n_models):
        print('Training model {}'.format(i))
        # init model
        deep_prop_model = EncoderBlock(args['model'], n_experiments)
        model = DeepClassifier(deep_prop_model).to(device)

        # init train
        optimizer = get_optimizer(args['train']['optimizer'])
        optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=args['train']['learning_rate'])
        intermediate_loss_type = get_loss_function(args['train']['intermediate_loss_type'],
                                                   focal_gamma=args['train']['focal_gamma'])
        trainer = ClassifierTrainer(args['train']['max_num_epochs'], criteria=nn.CrossEntropyLoss(reduction='sum'), intermediate_criteria=intermediate_loss_type,
                                    intermediate_loss_weight=args['train']['intermediate_loss_weight'],
                                    optimizer=optimizer, eval_metric=None, eval_interval=args['train']['eval_interval'], device=device)
        # train
        train_stats, best_model = trainer.train(train_loader=train_loader, eval_loader=val_loader, model=model,
                                                max_evals_no_improvement=args['train']['max_evals_no_imp'])
        models_list.append(best_model)
        train_stats_list.append(train_stats)

    best_model_idx = np.argmax([x['best_auc'] for x in train_stats_list])
    best_model = models_list[best_model_idx]
    train_stats = train_stats_list[best_model_idx]
    if len(test_dataset):
        test_results_dict = \
            trainer.eval_by_source(best_model, test_loader)
        print(test_results_dict)
    else:
        test_results_dict = {}

    results_dict = {'train_stats': train_stats, 'test_stats': test_results_dict, 'n_experiments': n_experiments,
                    'normalization_constants_dicts': {'samples': normalization_constants_dict, 'degrees': degree_normalization_constants}}
    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))
    save_model(path.join(output_file_path, 'model'), best_model)

if __name__ == '__main__':
    n_models = 1
    input_type = 'AML'
    n_exp = 5
    split = [0.8, 0.2, 0]
    interaction_type = sorted(['KPI'])
    device = None
    prop_scores_filename = input_type + '_' + '_'.join(interaction_type) + '_{}'.format(n_exp)

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str, help='name of experiment type(AML, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int, help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', dest='save_prop_scores',  action='store_true', default=False, help='Whether to save propagation scores')
    parser.add_argument('-l', '--load_prop', dest='load_prop_scores',  action='store_true', default=False, help='Whether to load prop scores')
    parser.add_argument('-sp', '--split', dest='train_val_test_split',  nargs=3, help='[train, val, test] sums to 1', default=split, type=float)
    parser.add_argument('-d', '--device', type=str, help='cpu or gpu number',  default=device)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', nargs='*', type=str,
                        help='KPI/STKE', default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('--n_models', dest='n_models', type=int,
                        help='number_of_models_to_train', default=n_models)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    args = parser.parse_args()

    run(args)

