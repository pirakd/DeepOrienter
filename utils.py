import numpy as np
import networkx as nx
import math
import scipy
from os import path
from datetime import datetime
import pickle
import json
import pandas as pd
import torch
from torch.optim import Adam, AdamW
from deep_learning.deep_utils import FocalLoss
from tqdm import tqdm
import sys
import copy
import argparse


def balance_dataset(graph, directed_interactions, rng):
    sources = directed_interactions['source'].unique()
    directed_interactions = directed_interactions[directed_interactions.index.get_level_values(0).isin(graph) & directed_interactions.index.get_level_values(1).isin(graph)]
    degree = dict(graph.degree(weight='edge_score'))

    balanced_interactions = pd.DataFrame()
    for source in sources:
        sliced_interactions = directed_interactions[directed_interactions['source'] == source]
        source_more_central = np.array([degree[s] > degree[t] for s, t in sliced_interactions.index])
        larger_indexes = np.nonzero(source_more_central)[0]
        smaller_indexes = np.nonzero(1-source_more_central)[0]
        if larger_indexes.size > smaller_indexes.size:
            larger_indexes = rng.choice(larger_indexes, smaller_indexes.size, replace=False)
        else:
            smaller_indexes = rng.choice(smaller_indexes, larger_indexes.size, replace=False)
        balanced_interactions = pd.concat([balanced_interactions, sliced_interactions.iloc[np.sort(np.hstack([smaller_indexes, larger_indexes]))]])

    return balanced_interactions


def read_network(network_filename, translator):
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2], index_col=[0, 1]).rename(
        columns={2: 'edge_score'})
    if translator:
        gene_ids = set(network.index.get_level_values(0)).union(set(network.index.get_level_values(1)))
        up_do_date_ids = translator.translate(gene_ids, 'entrez_id', 'entrez_id')
        network.rename(index=up_do_date_ids, inplace=True)

    network_index = np.sort(np.array([list(x) for x in network.index]), axis=1)
    network.index = pd.MultiIndex.from_tuples([(network_index[x, 0], network_index[x, 1])for x in
                                               range(network_index.shape[0])], names=[0, 1])

    # average same index entries
    network = network.groupby(level=[0, 1]).mean()

    return network


def read_directed_interactions(directed_interactions_folder, directed_interaction_filename, gene_translator):
    if isinstance(directed_interaction_filename, str):
        directed_interaction_filename = [directed_interaction_filename]
    directed_interaction_filename = sorted(directed_interaction_filename)
    all_interactions = pd.DataFrame()
    for name in directed_interaction_filename:
        directed_interactions = pd.read_table(path.join(directed_interactions_folder, name))
        directed_interactions['source'] = name
        # remove self edges
        genes = pd.unique(directed_interactions[['from', 'to']].values.ravel())

        if gene_translator is not None:
            if isinstance( directed_interactions['from'][0], str): # if genes in symbol format
                translation_dict = gene_translator.translate(genes, 'symbol', 'entrez_id')
            else: # if genes in entrez id format
                translation_dict = gene_translator.translate(genes, 'entrez_id', 'entrez_id')

            has_translation = directed_interactions['from'].isin(translation_dict) & directed_interactions['to'].isin(translation_dict)
            directed_interactions = directed_interactions[has_translation]
            directed_interactions.replace(translation_dict, inplace=True)


        directed_interactions = directed_interactions[directed_interactions['from'] != (directed_interactions['to'])]
        all_interactions = pd.concat((all_interactions, directed_interactions))


    all_interactions['edge_score'] = 0.8
    all_interactions.index = pd.MultiIndex.from_arrays(all_interactions[['from', 'to']].values.T)

    all_interactions = all_interactions[~all_interactions.duplicated(subset=['from','to'], keep='first')]

    all_interaction_set = set(list(all_interactions.index))
    all_interaction_flipped_set = set([(x[1], x[0]) for x in all_interaction_set])
    interactions_to_delete = list(all_interaction_flipped_set.intersection(all_interaction_set))
    interactions_to_delete = interactions_to_delete + [(x[1], x[0]) for x in interactions_to_delete]
    all_interactions = all_interactions.drop(interactions_to_delete)
    return all_interactions[['source', 'edge_score']]


def read_priors(sources_filename, terminals_filename, translator=None):
    source_priors = pd.read_table(sources_filename, header=None).groupby(0)[1].apply(set)
    terminal_priors = pd.read_table(terminals_filename, header=None).groupby(0)[1].apply(set)

    if translator:
        def translate(gene_set):
            translated_genes = translator.translate(gene_set, 'entrez_id', 'entrez_id')
            filtered_translated_genes = set([translated_genes[gene] for gene in gene_set if gene in translated_genes])
            return filtered_translated_genes
        source_priors = source_priors.apply(translate)
        terminal_priors = terminal_priors.apply(translate)

    # remove empty entreis
    source_priors = source_priors[source_priors.apply(lambda x:len(x)>0) > 0].to_dict()
    terminal_priors = terminal_priors[terminal_priors.apply(lambda x:len(x)>0) > 0].to_dict()
    return source_priors, terminal_priors


def read_data(network_filename, directed_interaction_filename, sources_filename, terminals_filename, n_exp,
              max_set_size, rng, translate_genes=True, is_balance_dataset=True):
    # set paths
    root_path = get_root_path()
    input_file = path.join(root_path, 'input')
    network_file_path = path.join(input_file, 'networks', network_filename)
    directed_interaction_folder = path.join(input_file, 'directed_interactions')
    sources_file_path = path.join(input_file, 'priors', sources_filename)
    terminals_file_path = path.join(input_file, 'priors', terminals_filename)

    # load gene translator
    if translate_genes:
        from gene_name_translator.gene_translator import GeneTranslator
        translator = GeneTranslator(verbosity=False)
        translator.load_dictionary()
    else:
        translator=None

    # do all the reading stuff
    network = read_network(network_file_path, translator)
    directed_interactions = read_directed_interactions(directed_interaction_folder, directed_interaction_filename, translator)

    sorted_interaction_df = copy.deepcopy(directed_interactions)
    sorted_interaction_index = np.sort(np.array([list(x) for x in directed_interactions.index]), axis=1)
    sorted_interaction_df.index = pd.MultiIndex.from_tuples([(sorted_interaction_index[x, 0], sorted_interaction_index[x, 1])for x in
                                               range(sorted_interaction_index.shape[0])], names=[0, 1])
    sorted_interaction_df = sorted_interaction_df[~sorted_interaction_df.index.duplicated()]
    merged_network =\
        pd.concat([network.drop(sorted_interaction_df.index.intersection(network.index)), sorted_interaction_df])
    merged_graph = nx.from_pandas_edgelist(merged_network.reset_index(), 0, 1, 'edge_score')
    if is_balance_dataset:
        directed_interactions = balance_dataset(merged_graph, directed_interactions, rng)
    sources, terminals = read_priors(sources_file_path, terminals_file_path, translator)

    # constrain to network's genes
    unique_genes = set(merged_graph.nodes)
    sources = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in sources.items()}
    terminals = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in terminals.items()}

    # filter large sets
    sources = {exp_name: values for exp_name, values in sources.items() if (0 < len(values) <= max_set_size )}
    terminals = {exp_name: values for exp_name, values in terminals.items() if (0 <len(values) <= max_set_size)}

    # filter experiments that do not have a source and a terminal set
    filtered_experiments = sorted(sources.keys() & terminals.keys())

    # choose only a subset of the experiments
    if n_exp == 0:
        pass
    elif isinstance(n_exp, int):
        filtered_experiments = filtered_experiments[:n_exp]
    else:
        assert 0, 'Wrong input in args[data][n_experiments]'
    sources = {exp_name: sources[exp_name] for exp_name in filtered_experiments}
    terminals = {exp_name: terminals[exp_name] for exp_name in filtered_experiments}
    id_to_degree = dict(merged_graph.degree(weight='edge_score'))

    return merged_network, directed_interactions, sources, terminals, id_to_degree


def gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list, calc_normalization_constants=True):
    root_path = get_root_path()
    if args['data']['load_prop_scores']:
        scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['prop_scores_filename'])
        scores_dict = load_pickle(scores_file_path)
        propagation_scores = scores_dict['propagation_scores']
        row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
        normalization_constants_dict = scores_dict['normalization_constants']
        assert scores_dict['data_args']['random_seed'] == args['data']['random_seed'], 'random seed of loaded data does not much current one'
    else:
        propagation_scores, row_id_to_idx, col_id_to_idx = generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep,
                                                                      args['propagation']['alpha'], args['propagation']['n_iterations'],
                                                                      args['propagation']['eps'])
        sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
        terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
        pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
        if calc_normalization_constants:
            normalization_constants_dict = get_normalization_constants(pairs_indexes, sources_indexes, terminals_indexes,
                                                                       propagation_scores, args['data']['normalization_method'])
        else:
            normalization_constants_dict = None
        if args['data']['save_prop_scores']:
            save_propagation_score(propagation_scores, normalization_constants_dict, row_id_to_idx, col_id_to_idx,
                                   args['propagation'], args['data'],  args['data']['prop_scores_filename'])
    return propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict

def generate_similarity_matrix(graph, propagate_alpha):
    genes = sorted(graph.nodes)
    matrix = nx.to_scipy_sparse_matrix(graph, genes, weight='edge_score')
    norm_matrix = scipy.sparse.diags(1 / np.sqrt(matrix.sum(0).A1))
    matrix = norm_matrix * matrix * norm_matrix
    return propagate_alpha * matrix, genes


def propagate(seeds, matrix, gene_indexes, num_genes, propagate_alpha, propagate_iterations, propagate_epsilon):
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - propagate_alpha) * F_t

    for _ in range(propagate_iterations):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < propagate_epsilon:
            break
    return F_t


def generate_propagate_data(network, propagate_alpha):
    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    matrix, genes = generate_similarity_matrix(graph, propagate_alpha)
    num_genes = len(genes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    return gene_indexes, matrix, num_genes


def generate_feature_columns(network, sources, terminals, indexes_to_keep, propagate_alpha, propagate_iterations, propagation_epsilon):
    gene_indexes, matrix, num_genes = generate_propagate_data(network, propagate_alpha)
    gene1_indexes, gene2_indexes = map(lambda x: x, zip(*[[(gene_indexes[gene]) for gene in pair] for pair in network.index]))
    experiments = sources.keys()

    def generate_column(experiment):
        source_scores = np.array([propagate([s], matrix, gene_indexes, num_genes, propagate_alpha, propagate_iterations,
                                            propagation_epsilon) for s in sources[experiment]]).T
        terminal_scores = np.array([propagate([t], matrix, gene_indexes, num_genes,  propagate_alpha, propagate_iterations,
                                            propagation_epsilon) for t in terminals[experiment]]).T
        source_gene_1, source_gene_2 = [source_scores[gene1_indexes, :], source_scores[gene2_indexes, :]]
        source_features = np.concatenate([source_gene_1[..., np.newaxis],  source_gene_2[..., np.newaxis,]], axis=2)[indexes_to_keep, ...]
        terminal_gene_1, terminal_gene_2 = [terminal_scores[gene1_indexes, :], terminal_scores[gene2_indexes, :]]
        terminal_features = np.concatenate([terminal_gene_1[..., np.newaxis],  terminal_gene_2[..., np.newaxis]], axis=2)[indexes_to_keep, ...]

        return source_features, terminal_features

    source_features, terminal_features = [], []
    for experiment in experiments:
        curr_source_features, curr_terminal_features = generate_column(experiment)
        source_features.append(curr_source_features)
        terminal_features.append(curr_terminal_features)

    return source_features, terminal_features


def generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep, propagate_alpha, propagate_iterations, propagation_epsilon):
    all_source_terminal_genes = sorted(list(set.union(set.union(*sources.values()), set.union(*terminals.values()))))
    gene_id_to_idx, matrix, num_genes = generate_propagate_data(network, propagate_alpha)
    gene_idx_to_id = {xx:x for x,xx in gene_id_to_idx.items()}

    propagation_scores = np.array([propagate([s], matrix, gene_id_to_idx, num_genes, propagate_alpha, propagate_iterations,
                                        propagation_epsilon) for s in tqdm(all_source_terminal_genes,
                                                                           desc='propagating scores',
                                                                           total=len(all_source_terminal_genes))])
    if genes_ids_to_keep:
        genes_idxs_to_keep = [gene_id_to_idx[id] for id in genes_ids_to_keep]
        propagation_scores = propagation_scores[:, genes_idxs_to_keep]
        col_id_to_idx = {gene_idx_to_id[idx]: i for i,idx in enumerate(genes_idxs_to_keep)}
    else:
        col_id_to_idx = {xx:x for x,xx in gene_idx_to_id.items()}

    row_id_to_idx = {id:i for i, id in enumerate(all_source_terminal_genes)}
    return propagation_scores, row_id_to_idx, col_id_to_idx


def normalize_features(source_features, terminal_features, eps=1e-8):
    source_array =[]
    terminal_array = []
    for arr_idx in range(len(source_features)):
        source_array.append(source_features[arr_idx].ravel())
        terminal_array.append(terminal_features[arr_idx].ravel())
    source_array, terminal_array = np.hstack(source_array), np.hstack(terminal_array)
    source_mean, terminal_mean = np.mean(source_array), np.mean(terminal_array)
    source_std, terminal_std = np.std(source_array), np.std(terminal_array)
    for arr_idx in range(len(source_features)):
        source_features[arr_idx] = ((source_features[arr_idx] - source_mean) / source_std) +eps
        terminal_features[arr_idx] = ((terminal_features[arr_idx] - terminal_mean) / terminal_std) +eps

    return source_features, terminal_features

def get_dataset_mean_std(pairs_indexes, source_indexes, terminal_indexes, propagation_scores):
    """
    Acording to Welford's algorithm for calculating variance, See attached PDF for derivation.
    """
    total_elements = 0
    total_mean, total_S = 0, 0
    total_examples = len(pairs_indexes) * len(source_indexes)
    max_num_examples = 25000
    total_examples = len(pairs_indexes) * len(source_indexes)
    p_sample = np.minimum(1, max_num_examples/total_examples)

    sampled_idx = np.nonzero(np.random.binomial(1, p_sample, int(p_sample * total_examples)))[0]
    pair_idxs = (sampled_idx//len(source_indexes)).astype(int)
    exp_idx = np.mod(sampled_idx, len(source_indexes))

    for idx in range(len(sampled_idx)):
            source_feature = propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][source_indexes[exp_idx[idx]],
                                    :].ravel()
            terminal_feature = propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][source_indexes[exp_idx[idx]],
                                    :].ravel()

            all_features = np.concatenate([source_feature,terminal_feature])

            num_new_elements =  len(all_features)
            total_elements += num_new_elements

            total_delta_mean = all_features - total_mean
            total_mean += np.sum(total_delta_mean) / total_elements
            total_S += np.sum((all_features - total_mean) * (total_delta_mean))

    total_std = np.sqrt(total_S/(total_elements-1))
    return {'mean':total_mean, 'std':total_std}

def get_power_transform_lambda(pairs_indexes, source_indexes, terminal_indexes, propagation_scores):
    from sklearn.preprocessing import PowerTransformer
    max_num_examples = 25000
    total_examples = len(pairs_indexes) * len(source_indexes)
    p_sample = np.minimum(1, max_num_examples/total_examples)

    sampled_idx = np.nonzero(np.random.binomial(1, p_sample, int(p_sample * total_examples)))[0]
    pair_idxs = (sampled_idx//len(source_indexes)).astype(int)
    exp_idx = np.mod(sampled_idx, len(source_indexes))
    sampled_elements = []
    for idx in range(len(sampled_idx)):
        sampled_elements.append(propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][source_indexes[exp_idx[idx]], :].ravel().tolist())
        sampled_elements.append(propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][terminal_indexes[exp_idx[idx]], :].ravel().tolist())

    sampled_elements = np.array([x for xx in sampled_elements for x in xx])
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = pt.fit_transform(sampled_elements[:, np.newaxis])
    mean = np.mean(transformed)
    std = np.std(transformed)
    return {'lmbda':pt.lambdas_[0], 'mean': mean, 'std':std}


def get_normalization_constants(pairs_indexes, source_indexes, terminal_indexes, propagation_scores, normalization_method):
    if normalization_method == 'standard':
        return get_dataset_mean_std(pairs_indexes, source_indexes, terminal_indexes, propagation_scores)
    elif normalization_method == 'power':
        return get_power_transform_lambda(pairs_indexes, source_indexes, terminal_indexes, propagation_scores)
    else:
        assert 0, '{} is not a valid normalization method name'.format(normalization_method)


def train_test_split(split_type ,n_samples, train_test_ratio,random_state:np.random.RandomState=None, directed_interactions=None, ):
    if split_type == 'normal':
        if random_state:
            split = random_state.choice([0, 1, 2], n_samples, replace=True, p=train_test_ratio )
        else:
            split = np.random.choice([0, 1, 2], n_samples, replace=True, p=train_test_ratio, )
        train_indexes = np.nonzero(split == 0)[0]
        val_indexes = np.nonzero(split == 1)[0]
        test_indexes = np.nonzero(split == 2)[0]
    elif split_type == 'harsh':

        interactions = np.array([list(directed_interactions[i]) for i in range(directed_interactions.shape[0])])
        u_genes, _, _, u_gene_counts  = np.unique(interactions[:,0], return_index=True, return_inverse=True, return_counts=True)

        sorted_indexes = np.argsort(u_gene_counts)[::-1]
        sorted_counts = u_gene_counts[sorted_indexes]
        sorted_u_genes = u_genes[sorted_indexes]

        gene_count = np.array([0, 0, 0])
        ratios = 1/(np.array(train_test_ratio)+1e-12)
        gene_piles = [[],[],[]]
        for i in range(len(sorted_counts)):
            n_added_genes = sorted_counts[i]
            pile_index_to_add = np.argmin((gene_count+n_added_genes)*ratios)
            gene_piles[pile_index_to_add].append(sorted_u_genes[i])
            gene_count[pile_index_to_add] += n_added_genes

        train_indexes = np.concatenate([np.nonzero(interactions[:,0]== gene_id)[0] for gene_id in gene_piles[0]]) if len(gene_piles[0])!=0 else list()
        val_indexes = np.concatenate([np.nonzero(interactions[:,0] == gene_id)[0] for gene_id in gene_piles[1]]) if len(gene_piles[1])!=0 else list()
        test_indexes = np.concatenate([np.nonzero(interactions[:,0]== gene_id)[0] for gene_id in gene_piles[2]]) if len(gene_piles[2])!=0 else list()
    else:
        assert 0, '{} is not a valid split type'.format(split_type)
    return train_indexes, val_indexes, test_indexes


def get_pulling_func(pulling_func_name):
    from deep_learning.deep_utils import MaskedSum, MaskedMean

    if pulling_func_name == 'mean':
        return MaskedMean
    elif pulling_func_name == 'sum':
        return MaskedSum
    else:
        assert 0, '{} is not a vallid pulling operation function name'.format(pulling_func_name)


def get_loss_function(loss_name, **kwargs):
    if loss_name == 'BCE':
        return torch.nn.BCEWithLogitsLoss(reduction='sum')
    elif loss_name == 'FOCAL':
        return FocalLoss(gamma=kwargs['focal_gamma'])
    else:
        assert 0, '{} is not a valid loss name'.format(loss_name)


def get_optimizer(optimizer_name):
    if optimizer_name == 'ADAMW':
        return AdamW
    elif optimizer_name == 'ADAM':
        return Adam
    else:
        assert 0, '{} is not a valid optimizer'.format(optimizer_name)

def get_root_path():
    return path.dirname(path.realpath(__file__))


def get_time():
    return datetime.today().strftime('%d_%m_%Y__%H_%M_%S_%f')[:-3]


def log_results(output_path, args, results_dict, model=None):
    with open(path.join(output_path, 'results'), 'w') as f:
        json.dump(results_dict, f, indent=4, separators=(',', ': '))
    with open(path.join(output_path, 'args'), 'w') as f:
        json.dump(args, f, indent=4, separators=(',', ': '))
    if model:
        save_model(path.join(output_path, 'model'), model)


def save_propagation_score(propagation_scores, normalization_constants,
                           row_id_to_idx, col_id_to_idx, propagation_args, data_args, filename):
    save_dir = path.join(get_root_path(), 'input', 'propagation_scores', filename)
    save_dict = {'propagation_args': propagation_args, 'row_id_to_idx': row_id_to_idx, 'col_id_to_idx': col_id_to_idx,
                 'propagation_scores': propagation_scores, 'normalization_constants': normalization_constants,
                 'data_args': data_args, 'data': get_time()}
    save_pickle(save_dict, save_dir)
    return save_dir


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
    from deep_learning.models import EncoderBlock, DeepClassifier
    state_dict = torch.load(path, map_location=device)
    n_experiments = state_dict['classifier.0.weight'].shape[1]
    deep_prop_model = EncoderBlock(args['model'], n_experiments)
    model = DeepClassifier(deep_prop_model)
    model.load_state_dict(state_dict)
    return model


def redirect_output(path):
    sys.stdout = Logger(path, sys.stdout)
    sys.stderr = Logger(path, sys.stderr)

class Logger(object):
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
        # you might want to specify some extra behavior here.
        pass


def generate_partially_directed_similarity_matrix(graph, prob_ratios, alpha):
    genes = sorted(graph.nodes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    matrix = nx.to_scipy_sparse_matrix(graph, genes, weight='edge_score')
    for pair, ratio in prob_ratios.items():
        head, tail = pair[0], pair[1]
        pair_indexes = gene_indexes[head], gene_indexes[tail]

        #  remove opposite direction, row=to, column=from
        # if edge (1,4) is directed then we zeroise index [1,4]
        matrix[pair_indexes[0], pair_indexes[1]] = matrix[pair_indexes[0], pair_indexes[1]] * ratio

    norm_matrix = scipy.sparse.diags(1 / matrix.sum(0).A1)
    sim_matrix = matrix * norm_matrix
    sim_matrix.data[np.isnan(sim_matrix.data)] = 0.0
    return alpha*sim_matrix, genes


def generate_directed_similarity_matrix(graph, directed_edge_list, alpha):
    genes = sorted(graph.nodes)
    gene_indexes = dict([(gene, index) for (index, gene) in enumerate(genes)])
    matrix = nx.to_scipy_sparse_matrix(graph, genes, weight='edge_score')
    for directed_edge in directed_edge_list:
        head, tail = directed_edge[0], directed_edge[1]
        pair_indexes = gene_indexes[head], gene_indexes[tail]

        #  remove opposite direction, row=to, column=from
        # if edge (1,4) is directed then we zeroise index [1,4]
        matrix[pair_indexes[0], pair_indexes[1]] = 0

    norm_matrix = scipy.sparse.diags(1 / matrix.sum(0).A1)
    sim_matrix = matrix * norm_matrix
    sim_matrix.data[np.isnan(sim_matrix.data)] = 0.0
    return alpha*sim_matrix, genes


def propagate_directed_network(undirected_network, directed_edges, sources, args):
    propagate_alpha = args['propagation']['alpha']
    propagation_iterations = args['propagation']['n_iterations']
    propagation_epsilon = args['propagation']['eps']
    graph = nx.from_pandas_edgelist(undirected_network.reset_index(), 0, 1, 'edge_score')
    directed_similarity_matrix, genes = generate_directed_similarity_matrix(graph, directed_edges, args['propagation']['alpha'])
    gene_id_to_idx = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_idx_to_id = {xx: x for x,xx in gene_id_to_idx.items()}
    num_genes = len(gene_idx_to_id)
    propagation_scores = []
    for source in tqdm(sources.values(), total=len(sources), desc= 'propagating scores'):
        propagation_scores.append(np.array(propagate([s for s in source],
                                                     directed_similarity_matrix, gene_id_to_idx,
                                                     num_genes, propagate_alpha, propagation_iterations,
                                                     propagation_epsilon)))

    col_id_to_idx = {xx:x for x,xx in gene_idx_to_id.items()}
    return propagation_scores, col_id_to_idx



def propagate_partially_directed_network(undirected_network, prob_ratios, sources, args):
    propagate_alpha = args['propagation']['alpha']
    propagation_iterations = args['propagation']['n_iterations']
    propagation_epsilon = args['propagation']['eps']
    graph = nx.from_pandas_edgelist(undirected_network.reset_index(), 0, 1, 'edge_score')
    directed_similarity_matrix, genes = generate_partially_directed_similarity_matrix(graph, prob_ratios, args['propagation']['alpha'])
    gene_id_to_idx = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_idx_to_id = {xx: x for x,xx in gene_id_to_idx.items()}
    num_genes = len(gene_idx_to_id)
    propagation_scores = []
    for source in tqdm(sources.values(), total=len(sources), desc= 'propagating scores'):
        propagation_scores.append(np.array(propagate([s for s in source],
                                                     directed_similarity_matrix, gene_id_to_idx,
                                                     num_genes, propagate_alpha, propagation_iterations,
                                                     propagation_epsilon)))

    col_id_to_idx = {xx:x for x,xx in gene_idx_to_id.items()}
    return propagation_scores, col_id_to_idx

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')