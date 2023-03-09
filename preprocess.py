import pandas as pd
import numpy as np
from utils import get_root_path, save_propagation_score, load_pickle
from os import path
import networkx as nx
import copy
import scipy
from tqdm import tqdm
import math


def _balance_dataset(graph, directed_interactions, rng):
    """
    A balanced ineraction dataset is one that contains an equal number of high-to-low-degree to low-
    to-high degree.
    this method receives an unbalanced dataset and returns a balanced version of it.
    :param graph: neworkx.Graph
    :param directed_interactions: pandas dataframe.
    :param rng: np.random.RandomState
    :return: pandas dataframe
    """
    sources = directed_interactions['source'].unique()
    directed_interactions = \
        directed_interactions[directed_interactions.index.get_level_values(0).isin(graph)
                              & directed_interactions.index.get_level_values(1).isin(graph)]
    degree = dict(graph.degree(weight='edge_score'))

    balanced_interactions = pd.DataFrame()
    for source in sources:
        sliced_interactions = directed_interactions[directed_interactions['source'] == source]
        source_more_central = np.array([degree[s] > degree[t] for s, t in sliced_interactions.index])
        larger_indexes = np.nonzero(source_more_central)[0]
        smaller_indexes = np.nonzero(1 - source_more_central)[0]
        if larger_indexes.size > smaller_indexes.size:
            larger_indexes = rng.choice(larger_indexes, smaller_indexes.size, replace=False)
        else:
            smaller_indexes = rng.choice(smaller_indexes, larger_indexes.size, replace=False)
        balanced_interactions = \
            pd.concat([balanced_interactions, sliced_interactions.iloc[np.sort(np.hstack([smaller_indexes,
                                                                                          larger_indexes]))]])
    return balanced_interactions


def read_network(network_filename, translator):
    """

    :param network_filename: path to network file
    :param translator: GeneTranslator object.
    :return: pandas.DataFrame. network with genes in entrez id format.
    """
    network = pd.read_table(network_filename, header=None, usecols=[0, 1, 2], index_col=[0, 1]).rename(
        columns={2: 'edge_score'})
    if translator:
        gene_ids = set(network.index.get_level_values(0)).union(set(network.index.get_level_values(1)))
        up_do_date_ids = translator.translate(gene_ids, 'entrez_id', 'entrez_id')
        network.rename(index=up_do_date_ids, inplace=True)

    network_index = np.sort(np.array([list(x) for x in network.index]), axis=1)
    network.index = pd.MultiIndex.from_tuples([(network_index[x, 0], network_index[x, 1]) for x in
                                               range(network_index.shape[0])], names=[0, 1])
    network = network.groupby(level=[0, 1]).mean()
    return network


def read_directed_interactions(directed_interactions_folder, directed_interaction_filename, gene_translator):
    """
    Reads and combine several directed interaction datasets into one.

    :param directed_interactions_folder: path for directed interaction datasets folder
    :param directed_interaction_filename: directed interaction filenames to include
    :param gene_translator: GeneTranslator object
    :return: pandas.DataFrame.
    """
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
            if isinstance(directed_interactions['from'][0], str):  # if genes in symbol format
                translation_dict = gene_translator.translate(genes, 'symbol', 'entrez_id')
            else:  # if genes in entrez id format
                translation_dict = gene_translator.translate(genes, 'entrez_id', 'entrez_id')
            has_translation =\
                directed_interactions['from'].isin(translation_dict) & \
                directed_interactions['to'].isin(translation_dict)
            directed_interactions = directed_interactions[has_translation]
            directed_interactions.replace(translation_dict, inplace=True)

        directed_interactions = directed_interactions[directed_interactions['from'] != (directed_interactions['to'])]
        all_interactions = pd.concat((all_interactions, directed_interactions))

    all_interactions['edge_score'] = 0.8
    all_interactions.index = pd.MultiIndex.from_arrays(all_interactions[['from', 'to']].values.T)
    all_interactions = all_interactions[~all_interactions.duplicated(subset=['from', 'to'], keep='first')]
    all_interaction_set = set(list(all_interactions.index))
    all_interaction_flipped_set = set([(x[1], x[0]) for x in all_interaction_set])
    interactions_to_delete = list(all_interaction_flipped_set.intersection(all_interaction_set))
    interactions_to_delete = interactions_to_delete + [(x[1], x[0]) for x in interactions_to_delete]
    all_interactions = all_interactions.drop(interactions_to_delete)
    return all_interactions[['source', 'edge_score']]


def read_priors(sources_filename, terminals_filename, translator=None):
    """
    read source and terminal set gene setes
    :param sources_filename: string.
    :param terminals_filename: string.
    :param translator: GeneTranslator object.
    :return: two dictionaries, {string: set}. each entry stands for {source_name: set_of_genes}
    """
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
    source_priors = source_priors[source_priors.apply(lambda x: len(x) > 0) > 0].to_dict()
    terminal_priors = terminal_priors[terminal_priors.apply(lambda x: len(x) > 0) > 0].to_dict()
    return source_priors, terminal_priors


def read_data(network_filename, directed_interaction_filename, sources_filename, terminals_filename, n_exp,
              max_set_size, rng, translate_genes=True, balance_dataset=True):
    """
    method to read all needed data for a run
    :param network_filename: string.
    :param directed_interaction_filename: string.
    :param sources_filename: string.
    :param terminals_filename: string.
    :param n_exp: int. number of experiments/patients to load.
    :param max_set_size:  int. maximum number of elements in a set.
    :param rng: np.random.RandomState
    :param translate_genes: GeneTranslator object.
    :param is_balance_dataset: bool. whether to balance interaction datasets.
    :return: networkx.Graph, pandas.Dataframe, dict, dict, dict
    """
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
        translator = None

    # do all the reading stuff
    network = read_network(network_file_path, translator)
    directed_interactions = read_directed_interactions(directed_interaction_folder, directed_interaction_filename,
                                                       translator)

    sorted_interaction_df = copy.deepcopy(directed_interactions)
    sorted_interaction_index = np.sort(np.array([list(x) for x in directed_interactions.index]), axis=1)
    sorted_interaction_df.index = pd.MultiIndex.from_tuples(
        [(sorted_interaction_index[x, 0], sorted_interaction_index[x, 1]) for x in
         range(sorted_interaction_index.shape[0])], names=[0, 1])
    sorted_interaction_df = sorted_interaction_df[~sorted_interaction_df.index.duplicated()]
    merged_network = \
        pd.concat([network.drop(sorted_interaction_df.index.intersection(network.index)), sorted_interaction_df])
    merged_graph = nx.from_pandas_edgelist(merged_network.reset_index(), 0, 1, 'edge_score')
    if balance_dataset:
        directed_interactions = _balance_dataset(merged_graph, directed_interactions, rng)
    sources, terminals = read_priors(sources_file_path, terminals_file_path, translator)

    # constrain to network's genes
    unique_genes = set(merged_graph.nodes)
    sources = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in sources.items()}
    terminals = {exp_name: set([gene for gene in genes if gene in unique_genes]) for exp_name, genes in
                 terminals.items()}

    # filter large sets
    sources = {exp_name: values for exp_name, values in sources.items() if (0 < len(values) <= max_set_size)}
    terminals = {exp_name: values for exp_name, values in terminals.items() if (0 < len(values) <= max_set_size)}

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


def gen_propagation_scores(args, network, sources, terminals, genes_ids_to_keep, directed_interactions_pairs_list,
                           calc_normalization_constants=True):
    """
    creates a 2d array matrix of propagation scores. rows and column indexes are according to returned mappings.
    only genes that appear in directed_interactions_pairs_list are kept.
    :param args: args dict as in presets.py
    :param network: pandas.Dataframe.
    :param sources: dict.
    :param terminals: dict.
    :param genes_ids_to_keep: iterable.
    :param directed_interactions_pairs_list: list of tuples.
    :param calc_normalization_constants: boolean
    :return: numpy.ndarray, , dict, dict, dict
    """
    root_path = get_root_path()
    if args['data']['load_prop_scores']:
        scores_file_path = path.join(root_path, 'input', 'propagation_scores', args['data']['prop_scores_filename'])
        scores_dict = load_pickle(scores_file_path)
        propagation_scores = scores_dict['propagation_scores']
        row_id_to_idx, col_id_to_idx = scores_dict['row_id_to_idx'], scores_dict['col_id_to_idx']
        normalization_constants_dict = scores_dict['normalization_constants']
        assert scores_dict['data_args']['random_seed'] == args['data'][
            'random_seed'], 'random seed of loaded data does not much current one'
    else:
        propagation_scores, row_id_to_idx, col_id_to_idx =\
            _generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep,
                                            args['propagation']['alpha'], args['propagation']['n_iterations'],
                                            args['propagation']['eps'])
        sources_indexes = [[row_id_to_idx[id] for id in set] for set in sources.values()]
        terminals_indexes = [[row_id_to_idx[id] for id in set] for set in terminals.values()]
        pairs_indexes = [(col_id_to_idx[pair[0]], col_id_to_idx[pair[1]]) for pair in directed_interactions_pairs_list]
        if calc_normalization_constants:
            normalization_constants_dict = get_normalization_constants(pairs_indexes, sources_indexes,
                                                                       terminals_indexes,
                                                                       propagation_scores,
                                                                       args['data']['normalization_method'])
        else:
            normalization_constants_dict = None
        if args['data']['save_prop_scores']:
            save_propagation_score(propagation_scores, normalization_constants_dict, row_id_to_idx, col_id_to_idx,
                                   args['propagation'], args['data'], args['data']['prop_scores_filename'])
    return propagation_scores, row_id_to_idx, col_id_to_idx, normalization_constants_dict


def _generate_similarity_matrix(graph, propagate_alpha):
    """
    generates a normalized similarity matrix for network propagation
    :param graph: networkx.Graph
    :param propagate_alpha: float. propagation parameter
    :return: scipy.sparse.csr_matrix
    """
    genes = sorted(graph.nodes)
    matrix = nx.to_scipy_sparse_matrix(graph, genes, weight='edge_score')
    norm_matrix = scipy.sparse.diags(1 / np.sqrt(matrix.sum(0).A1))
    matrix = norm_matrix * matrix * norm_matrix
    return propagate_alpha * matrix, genes


def propagate(seeds, matrix, gene_indexes, num_genes, propagate_alpha, propagate_iterations, propagate_epsilon):
    """
    core propagation routine
    :param seeds: iterable. genes to propagate from
    :param matrix: scipy.sparse.csr_matrix. normalized similarity matrix
    :param gene_indexes: dict. id to idx mapping for seeds
    :param num_genes: int. number of genes in the graph
    :param propagate_alpha: float. propagation parameter
    :param propagate_iterations: int. maximum number of iterations
    :param propagate_epsilon: float. convergance thresholds.
    :return:
    """
    F_t = np.zeros(num_genes)
    F_t[[gene_indexes[seed] for seed in seeds if seed in gene_indexes]] = 1
    Y = (1 - propagate_alpha) * F_t

    for _ in range(propagate_iterations):
        F_t_1 = F_t
        F_t = matrix.dot(F_t_1) + Y

        if math.sqrt(scipy.linalg.norm(F_t_1 - F_t)) < propagate_epsilon:
            break
    return F_t


def _generate_raw_propagation_scores(network, sources, terminals, genes_ids_to_keep, propagate_alpha,
                                    propagate_iterations, propagation_epsilon):
    """
    handles propagation routine.
    :param network:
    :param sources:
    :param terminals:
    :param genes_ids_to_keep:
    :param propagate_alpha:
    :param propagate_iterations:
    :param propagation_epsilon:
    :return:
    """
    all_source_terminal_genes = sorted(list(set.union(set.union(*sources.values()), set.union(*terminals.values()))))

    graph = nx.from_pandas_edgelist(network.reset_index(), 0, 1, 'edge_score')
    matrix, genes = _generate_similarity_matrix(graph, propagate_alpha)
    num_genes = len(genes)
    gene_id_to_idx = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_idx_to_id = {xx: x for x, xx in gene_id_to_idx.items()}
    propagation_scores = np.array(
        [propagate([s], matrix, gene_id_to_idx, num_genes, propagate_alpha, propagate_iterations,
                   propagation_epsilon) for s in tqdm(all_source_terminal_genes,
                                                      desc='propagating scores',
                                                      total=len(all_source_terminal_genes))])
    if genes_ids_to_keep:
        genes_idxs_to_keep = [gene_id_to_idx[id] for id in genes_ids_to_keep]
        propagation_scores = propagation_scores[:, genes_idxs_to_keep]
        col_id_to_idx = {gene_idx_to_id[idx]: i for i, idx in enumerate(genes_idxs_to_keep)}
    else:
        col_id_to_idx = {xx: x for x, xx in gene_idx_to_id.items()}

    row_id_to_idx = {id: i for i, id in enumerate(all_source_terminal_genes)}
    return propagation_scores, row_id_to_idx, col_id_to_idx


def get_normalization_constants(pairs_indexes, source_indexes, terminal_indexes, propagation_scores,
                                normalization_method):
    """
    return normalization constants according to normalization constants
    :param pairs_indexes: list of tuples.
    :param source_indexes: dict.
    :param terminal_indexes: dict.
    :param propagation_scores: np.ndarray
    :param normalization_method: string.
    :return: dict.

    """
    if normalization_method == 'standard':
        return get_dataset_mean_std(pairs_indexes, source_indexes, terminal_indexes, propagation_scores)
    elif normalization_method == 'power':
        return get_power_transform_lambda(pairs_indexes, source_indexes, terminal_indexes, propagation_scores)
    else:
        assert 0, '{} is not a valid normalization method name'.format(normalization_method)


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
    return alpha * sim_matrix, genes


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
    return alpha * sim_matrix, genes


def propagate_directed_network(undirected_network, directed_edges, sources, args):
    propagate_alpha = args['propagation']['alpha']
    propagation_iterations = args['propagation']['n_iterations']
    propagation_epsilon = args['propagation']['eps']
    graph = nx.from_pandas_edgelist(undirected_network.reset_index(), 0, 1, 'edge_score')
    directed_similarity_matrix, genes = generate_directed_similarity_matrix(graph, directed_edges,
                                                                            args['propagation']['alpha'])
    gene_id_to_idx = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_idx_to_id = {xx: x for x, xx in gene_id_to_idx.items()}
    num_genes = len(gene_idx_to_id)
    propagation_scores = []
    for source in tqdm(sources.values(), total=len(sources), desc='propagating scores'):
        propagation_scores.append(np.array(propagate([s for s in source],
                                                     directed_similarity_matrix, gene_id_to_idx,
                                                     num_genes, propagate_alpha, propagation_iterations,
                                                     propagation_epsilon)))

    col_id_to_idx = {xx: x for x, xx in gene_idx_to_id.items()}
    return propagation_scores, col_id_to_idx


def propagate_partially_directed_network(undirected_network, prob_ratios, sources, args):
    propagate_alpha = args['propagation']['alpha']
    propagation_iterations = args['propagation']['n_iterations']
    propagation_epsilon = args['propagation']['eps']
    graph = nx.from_pandas_edgelist(undirected_network.reset_index(), 0, 1, 'edge_score')
    directed_similarity_matrix, genes = generate_partially_directed_similarity_matrix(graph, prob_ratios,
                                                                                      args['propagation']['alpha'])
    gene_id_to_idx = dict([(gene, index) for (index, gene) in enumerate(genes)])
    gene_idx_to_id = {xx: x for x, xx in gene_id_to_idx.items()}
    num_genes = len(gene_idx_to_id)
    propagation_scores = []
    for source in tqdm(sources.values(), total=len(sources), desc='propagating scores'):
        propagation_scores.append(np.array(propagate([s for s in source],
                                                     directed_similarity_matrix, gene_id_to_idx,
                                                     num_genes, propagate_alpha, propagation_iterations,
                                                     propagation_epsilon)))

    col_id_to_idx = {xx: x for x, xx in gene_idx_to_id.items()}
    return propagation_scores, col_id_to_idx


def get_power_transform_lambda(pairs_indexes, source_indexes, terminal_indexes, propagation_scores):
    """
    Calculates Yeo-Johnson transforma constants
    Acording to Welford's algorithm for recursive calculation variance, See attached PDF for derivation.
    :param pairs_indexes: list of tuples.
    :param source_indexes: list of lists.
    :param terminal_indexes: list of lists.
    :param propagation_scores: np.ndarray.
    :return: dict.
    """
    from sklearn.preprocessing import PowerTransformer
    max_num_examples = 25000
    total_examples = len(pairs_indexes) * len(source_indexes)
    p_sample = np.minimum(1, max_num_examples / total_examples)

    sampled_idx = np.nonzero(np.random.binomial(1, p_sample, int(p_sample * total_examples)))[0]
    pair_idxs = (sampled_idx // len(source_indexes)).astype(int)
    exp_idx = np.mod(sampled_idx, len(source_indexes))
    sampled_elements = []
    for idx in range(len(sampled_idx)):
        sampled_elements.append(
            propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][source_indexes[exp_idx[idx]], :].ravel().tolist())
        sampled_elements.append(
            propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][terminal_indexes[exp_idx[idx]], :].ravel().tolist())

    sampled_elements = np.array([x for xx in sampled_elements for x in xx])
    pt = PowerTransformer(method='yeo-johnson', standardize=False)
    transformed = pt.fit_transform(sampled_elements[:, np.newaxis])
    mean = np.mean(transformed)
    std = np.std(transformed)
    return {'lmbda': pt.lambdas_[0], 'mean': mean, 'std': std}


def get_dataset_mean_std(pairs_indexes, source_indexes, terminal_indexes, propagation_scores):
    """
    Acording to Welford's algorithm for recursive calculation variance, See attached PDF for derivation.
    :param pairs_indexes: list of tuples.
    :param source_indexes: list of lists.
    :param terminal_indexes: list of lists.
    :param propagation_scores: np.ndarray.
    :return: dict.
    """
    total_elements = 0
    total_mean, total_S = 0, 0
    max_num_examples = 25000
    total_examples = len(pairs_indexes) * len(source_indexes)
    p_sample = np.minimum(1, max_num_examples / total_examples)
    sampled_idx = np.nonzero(np.random.binomial(1, p_sample, int(p_sample * total_examples)))[0]
    pair_idxs = (sampled_idx // len(source_indexes)).astype(int)
    exp_idx = np.mod(sampled_idx, len(source_indexes))

    for idx in range(len(sampled_idx)):
        source_feature = propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][source_indexes[exp_idx[idx]],
                         :].ravel()
        terminal_feature = propagation_scores[:, [pairs_indexes[pair_idxs[idx]]]][terminal_indexes[exp_idx[idx]],
                           :].ravel()

        all_features = np.concatenate([source_feature, terminal_feature])
        num_new_elements = len(all_features)
        total_elements += num_new_elements
        total_delta_mean = all_features - total_mean
        total_mean += np.sum(total_delta_mean) / total_elements
        total_S += np.sum((all_features - total_mean) * (total_delta_mean))

    total_std = np.sqrt(total_S / (total_elements - 1))
    return {'mean': total_mean, 'std': total_std}
