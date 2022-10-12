import numpy as np
import copy
import networkx as nx
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_recall_curve, auc


def count_sp_edges(network, sources, terminals):
    all_sources = np.array(list(set(sources.keys())), dtype=int)
    all_terminals = np.array(list(set(terminals.keys())),dtype=int)
    shortset_paths = {}
    edges_in_paths = defaultdict(set)
    edges_in_grouped_path = defaultdict(set)
    edge_count = defaultdict(int)
    edge_count_grouped = defaultdict(int)
    for p, pair in tqdm(enumerate(product(all_sources, all_terminals)), desc="Computing shortest paths", total=(len(all_sources)* len(all_terminals))):
        group = (sources[pair[0]], terminals[pair[1]])
        try:
            shortset_paths[pair] = nx.shortest_path(network, pair[0], pair[1])
            for n, node in enumerate(shortset_paths[pair][:-1]):
                edge = (shortset_paths[pair][n], shortset_paths[pair][n+1])
                edges_in_paths[pair].add(edge)
                edges_in_grouped_path[group].add(edge)
                edge_count[edge] += 1
        except:
            continue

    for group in edges_in_grouped_path.values():
        for pair in group:
            edge_count_grouped[pair] += 1

    return edge_count, edge_count_grouped

def generate_vinyagam_feature(network, edge_count, edge_count_grouped, samples):
    """
    :param source_features: list of experiments each of size [n_samples, n_sources, 2]
    :param terminal_featues: list of experiments each of size [n_samples, n_terminals, 2]
    :return: a numpy array of features of size [n_experiments, n_samples)
    """
    n_original_samples = len(samples)
    num_edges = np.sum(edge_count[x] for x in edge_count.keys())
    num_grouped_edges = np.sum(edge_count_grouped[x] for x in edge_count_grouped.keys())
    samples = list(samples) + [(x[1], x[0]) for x in samples]
    feature_1 = []
    feature_2 = []
    feature_3 = []
    feature_4 = []
    feature_5 = []
    feature_6 = []
    feature_7 = []
    feature_8 = []
    for pair in samples:
        #feature_1
        try:
            feature_1.append(edge_count[pair]/ (edge_count[pair]+ edge_count[(pair[1],pair[0])]))
        except:
            feature_1.append(0.5)

        #feature 2
        try:
            feature_2.append(edge_count_grouped[pair]/ (edge_count_grouped[pair]+ edge_count[(pair[1],pair[0])]))
        except:
            feature_2.append(0.5)

        #  feature 3 + feature 4
        feature_3.append(edge_count[pair]/num_edges)
        feature_4.append(edge_count_grouped[pair]/num_grouped_edges)

        #feautre 5 + feature 6
        M_icn, N_icn, M_ocn, N_ocn = 0, 0, 0, 0
        for neighbor in nx.common_neighbors(network, pair[0], pair[1]):
            M_icn += edge_count[(pair[0], neighbor)]
            N_icn +=  edge_count[(pair[0], neighbor)] + edge_count[(neighbor, pair[0])]
            M_ocn += edge_count[(pair[1], neighbor)]
            N_ocn += edge_count[(pair[1], neighbor)] + edge_count[(neighbor, pair[1])]
        try:
            feature_5.append(M_icn/N_icn)
        except:
            feature_5.append(0.5)
        try:
            feature_6.append(M_ocn/N_ocn)
        except:
            feature_6.append(0.5)

        #feautre 7 + feature 8
        M_ing, N_i, M_ong, N_o = 0, 0, 0, 0
        for neighbor in network.neighbors(pair[0]):
            M_ing += edge_count[(pair[0], neighbor)]
            N_i += edge_count[(neighbor, pair[0])] + edge_count[(pair[0], neighbor)]
        try:
            feature_7.append(M_ing/N_i)
        except:
            feature_7.append(0.5)

        for neighbor in network.neighbors(pair[1]):
            M_ong += edge_count[(pair[1], neighbor)]
            N_o += edge_count[(neighbor, pair[1])] + edge_count[(pair[1], neighbor)]
        try:
            feature_8.append(M_ong/N_o)
        except:
            feature_8.append(0.5)

    features = np.vstack([feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7, feature_8]).T

    labels = np.zeros(len(samples))
    labels[:n_original_samples] = 1

    return features, labels


def infer_vinayagam(train_features, train_labels, test_features, test_labels, source_types=None):
    model = GaussianNB()
    model.fit(train_features, train_labels)
    probs = model.predict_proba(test_features)

    type_output_dict = {'probs': [], 'labels': []}
    results_by_source_type = {}
    results_by_source_type['overall'] = copy.deepcopy(type_output_dict)
    results_by_source_type['overall']['probs'] = probs
    results_by_source_type['overall']['labels'] = test_labels
    results_by_source_type['overall']['acc'] = np.mean(np.argmax(probs, 1) == test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels,
                                                           probs[:, 1])
    results_by_source_type['overall']['auc'] = auc(recall, precision)
    if len(precision) == 2:
        results_by_source_type['overall']['auc'] = 0.5
        results_by_source_type['overall']['precision'] = [0.5, 0.5]

    if source_types is not None:
        unique_source_types = np.unique(source_types)
        source_types = np.concatenate([source_types, source_types])
        for source_type in unique_source_types:
            source_samples_idx = source_types == source_type
            results_by_source_type[source_type] = copy.deepcopy(type_output_dict)
            results_by_source_type[source_type]['probs'] = probs[source_samples_idx]
            results_by_source_type[source_type]['labels'] = test_labels[source_samples_idx]
            results_by_source_type[source_type]['acc'] = np.mean(np.argmax(probs[source_samples_idx], 1) == test_labels[source_samples_idx])
            precision, recall, thresholds = precision_recall_curve(test_labels[source_samples_idx], probs[source_samples_idx, 1])
            results_by_source_type[source_type]['auc'] = auc(recall, precision)

    return results_by_source_type, model

