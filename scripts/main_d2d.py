from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import path, makedirs
from utils import read_data, get_root_path, train_test_split, get_time, \
    gen_propagation_scores, redirect_output, str2bool
from D2D import eval_D2D, generate_D2D_features_from_propagation_scores
import numpy as np
from presets import example_preset
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import pickle

def run(sys_args):
    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    n_experiments = sys_args.n_experiments
    args = example_preset

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
    args['data']['prop_scores_filename'] = sys_args.prop_scores_filename
    args['train']['train_val_test_split'] = sys_args.train_val_test_split
    args['data']['directed_interactions_filename'] = sys_args.directed_interactions_filename
    args['data']['sources_filename'] = sources_filenmae_dict[sys_args.experiments_type]
    args['data']['terminals_filename'] = terminals_filenmae_dict[sys_args.experiments_type]
    args['data']['n_experiments'] = n_experiments
    print(json.dumps(args, indent=4))

    # data read and filtering
    rng = np.random.RandomState(args['data']['random_seed'])

    network, directed_interactions, sources, terminals, id_to_degree = \
        read_data(args['data']['network_filename'], args['data']['directed_interactions_filename'],
                  args['data']['sources_filename'], args['data']['terminals_filename'],
                  args['data']['n_experiments'], args['data']['max_set_size'], rng)
    n_experiments = len(sources)

    # merged_network = pandas.concat([directed_interactions, network.drop(directed_interactions.index & network.index)])
    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)
    genes_ids_to_keep = sorted(list(set([x for pair in directed_interactions_pairs_list for x in pair])))

    propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict = \
        gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list)

    train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list),
                                                                args['train']['train_val_test_split'],
                                                                random_state=rng)  # feature generation
    train_indexes = np.concatenate([train_indexes, val_indexes])
    d2d_train_indexes = np.concatenate([train_indexes, val_indexes])

    # d2d evaluation
    sources_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in sources.values()]
    terminals_indexes = [[row_id_to_idx[gene_id] for gene_id in gene_set] for gene_set in terminals.values()]
    pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
    features, deconstructed_features = generate_D2D_features_from_propagation_scores(propagation_scores,
                                                                                     pairs_indexes,
                                                                                     sources_indexes,
                                                                                     terminals_indexes)
    d2d_results_dict, d2d_model = eval_D2D(features[d2d_train_indexes], features[test_indexes],
                                       directed_interactions_source_type[test_indexes])
    d2d_stats = ({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                            d2d_results_dict.items()})

    results =  {'d2d': d2d_stats,
                     'n_experiments':n_experiments}
    print('Overall results: {}'.format(d2d_stats['overall']))

    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(results, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'd2d_models'), 'wb') as f:
        pickle.dump(d2d_model, f)


if __name__ == '__main__':
    input_type = 'AML'
    n_exp = 5
    split = [0.66, 0.14, 0.2]
    interaction_type = sorted(['KPI'])
    prop_scores_filename = input_type + '_' + '_'.join(interaction_type) + '_{}'.format(n_exp)

    parser = argparse.ArgumentParser()
    parser.add_argument('-ex', '--ex_type', dest='experiments_type', type=str,
                        help='name of experiment type(drug, colon, etc.)', default=input_type)
    parser.add_argument('-n,', '--n_exp', dest='n_experiments', type=int,
                        help='num of experiments used (0 for all)', default=n_exp)
    parser.add_argument('-s', '--save_prop', type=str2bool, dest='save_prop_scores', nargs='?', default=False,
                        help="whether to save computed propagation scores")
    parser.add_argument('-l', '--load_prop', type=str2bool, dest='load_prop_scores', nargs='?', default=True,
                        help="whether to load pre-computed propagation scores")
    parser.add_argument('-sp', '--split', dest='train_val_test_split', nargs=3, help='[train, val, test] sums to 1',
                        default=split, type=float)
    parser.add_argument('-in', '--inter_file', dest='directed_interactions_filename', nargs='*', type=str,
                        help='KPI/STKE', default=interaction_type)
    parser.add_argument('-p', '--prop_file', dest='prop_scores_filename', type=str,
                        help='Name of prop score file(save/load)', default=prop_scores_filename)
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    parser.add_argument('-d', dest='device', type=int, help='gpu number', default=None)
    args = parser.parse_args()

    run(args)
