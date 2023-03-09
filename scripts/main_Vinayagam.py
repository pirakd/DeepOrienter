from os import path
import sys
sys.path.append(path.dirname(path.dirname(path.realpath(__file__))))
from os import path, makedirs
from utils import get_root_path, train_test_split, get_time, redirect_output, str2bool
from preprocess import read_data
from Vinayagam import generate_vinyagam_feature, count_sp_edges, infer_vinayagam
import numpy as np
from presets import example_preset
import json
import argparse
from scripts.scripts_utils import sources_filenmae_dict, terminals_filenmae_dict
import pickle
import networkx as nx
import pandas as pd
from gene_name_translator.gene_translator import GeneTranslator


def run(sys_args):
    gene_translator = GeneTranslator(verbosity=True)
    gene_translator.load_dictionary()

    output_folder = 'output'
    output_file_path = path.join(get_root_path(), output_folder, path.basename(__file__).split('.')[0], get_time())
    makedirs(output_file_path, exist_ok=True)
    redirect_output(path.join(output_file_path, 'log'))

    n_experiments = sys_args.n_experiments
    args = example_preset

    args['data']['load_prop_scores'] = sys_args.load_prop_scores
    args['data']['save_prop_scores'] = sys_args.save_prop_scores
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

    transcription_factors_df = pd.read_csv(path.join(get_root_path(), 'input', 'other', 'transcription_factors.tsv'), sep='\t')
    receptors_df = pd.read_csv(path.join(get_root_path(), 'input', 'other', 'receptors.tsv'), sep='\t')
    transcription_factors_df['DBD'] = transcription_factors_df['DBD'].str.split(';').str[0]
    receptors_df['Classification'] = receptors_df['Classification']
    all_transcription_factors = transcription_factors_df['Symbol'].to_list()
    all_receptors = receptors_df['Entrez_ID'].to_list()
    symbol_to_entrez = gene_translator.translate(all_transcription_factors, 'symbol', 'entrez_id')
    entrez_to_entrez = gene_translator.translate(all_receptors, 'entrez_id', 'entrez_id')

    transcription_factors_df['Symbol'] = transcription_factors_df['Symbol'].apply(lambda x: symbol_to_entrez.get(x, None))
    receptors_df['Entrez_ID'] = receptors_df['Entrez_ID'].apply(lambda x: entrez_to_entrez.get(x, None))
    transcription_factors_df.rename(columns={'Symbol':'Entrez_ID'}, inplace=True)
    transcription_factors_df.dropna(inplace=True)
    tf_groups = transcription_factors_df.set_index('Entrez_ID').to_dict()['DBD']
    receptor_groups = receptors_df.set_index('Entrez_ID').to_dict()['Classification']
    transcription_factors_df.to_dict()
    network['edge_score'] = 1-network['edge_score']
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')

    directed_interactions_pairs_list = np.array(directed_interactions.index)
    directed_interactions_source_type = np.array(directed_interactions.source)

    counts, grouped_counts = count_sp_edges(graph, receptor_groups, tf_groups)

    train_indexes, val_indexes, test_indexes = train_test_split(len(directed_interactions_pairs_list),
                                                                args['train']['train_val_test_split'],
                                                                random_state=rng)  # feature generation
    train_indexes = np.concatenate([train_indexes, val_indexes])

    train_features, train_labels = generate_vinyagam_feature(graph, counts, grouped_counts, directed_interactions_pairs_list[train_indexes])
    test_features, test_labels = generate_vinyagam_feature(graph, counts, grouped_counts, directed_interactions_pairs_list[test_indexes])
    results_dict, model = infer_vinayagam(train_features, train_labels, test_features, test_labels, directed_interactions_source_type[test_indexes])

    vinayagam_stats= ({type: {x: xx for x, xx in values.items() if x in ['acc', 'auc']} for type, values in
                            results_dict.items()})

    print(vinayagam_stats)
    models = {'model':model}

    with open(path.join(output_file_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'results'), 'w') as f:
        json.dump(vinayagam_stats, f, indent=4, separators=(',', ': '))
    with open(path.join(output_file_path, 'vinayagam_model'), 'wb') as f:
        pickle.dump(models, f)


if __name__ == '__main__':
    input_type = 'AML'
    load_prop = False
    save_prop = False
    n_exp = 5
    split = [0.66, 0.14, 0.2]
    interaction_type = ['KPI']
    prop_scores_filename = None

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
    parser.add_argument('-w', dest='n_workers', type=int,
                        help='number of dataloader workers', default=0)
    parser.add_argument('-d', dest='device', type=int, help='gpu number', default=None)
    args = parser.parse_args()

    run(args)
