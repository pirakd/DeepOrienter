import numpy as np
from sklearn import linear_model
from sklearn.metrics import precision_recall_curve, auc
import copy

def generate_D2D_features_from_propagation_scores(propagation_scores, pairs_indexes, source_indexes, terminal_indexes):
    """
    :param source_features: list of experiments each of size [n_samples, n_sources, 2]
    :param terminal_featues: list of experiments each of size [n_samples, n_terminals, 2]
    :return: a numpy array of features of size [n_experiments, n_samples)
    """

    gene1, gene2 = map(lambda x: x, zip(*[[gene for gene in pair] for pair in pairs_indexes]))
    features = []
    deconstructed_features = []
    for exp_idx in range(len(source_indexes)):
        source_feature_1 = np.sum(propagation_scores[source_indexes[exp_idx], :][:, gene1],axis=0)
        source_feature_2 = np.sum(propagation_scores[source_indexes[exp_idx],:][:,gene2],axis=0)
        terminal_feature_1 = np.sum(propagation_scores[terminal_indexes[exp_idx],:][:,gene1],axis=0)
        terminal_feature_2 = np.sum(propagation_scores[terminal_indexes[exp_idx],:][:,gene2],axis=0)

        numerator = source_feature_1* terminal_feature_2
        denominator = source_feature_2 * terminal_feature_1
        features.append(numerator/denominator)
        deconstructed_features.append(np.array([source_feature_1, terminal_feature_1, source_feature_2,  terminal_feature_2 ]))
    features = np.array(features).T
    deconstructed_features = np.array(np.concatenate(deconstructed_features,axis=0)).T
    return features, deconstructed_features


def generate_D2D_features(source_features, terminal_featues):
    """
    :param source_features: list of experiments each of size [n_samples, n_sources, 2]
    :param terminal_featues: list of experiments each of size [n_samples, n_terminals, 2]
    :return: a numpy array of features of size [n_experiments, n_samples)
    """

    features = []
    for exp_idx in range(len(source_features)):
        source_sum = np.sum(source_features[exp_idx], axis=1)
        terminal_sum = np.sum(terminal_featues[exp_idx], axis=1)

        numerator = source_sum[:, 0] * terminal_sum[:, 1]
        denominator = source_sum[:, 1] * terminal_sum[:, 0]
        features.append(numerator/denominator)

    features = np.array(features).T
    return features


def eval_D2D(train_features, test_features, source_types=None):
    inverse_train_features = 1 / train_features
    inverse_test_features = 1 / test_features

    train_labels, test_labels = np.zeros(train_features.shape[0] * 2),  np.zeros(test_features.shape[0] * 2)
    train_labels[:train_features.shape[0]] = 1
    test_labels[:test_features.shape[0]] = 1

    train_features = np.vstack([train_features, inverse_train_features])
    test_features = np.vstack([test_features, inverse_test_features])

    # this was not mentioned in their paper but appeared in their git repository:
    # https://github.com/danasilv/Diffuse2Direct/blob/master/code/Diffuse2Direct.ipynb (cell 9)
    quantized_train_features = np.argsort(np.argsort(train_features, axis=1), axis=1)
    quantized_test_features = np.argsort(np.argsort(test_features, axis=1), axis=1)

    clf = linear_model.LogisticRegression(solver='liblinear', penalty='l1', C=0.01)
    clf.fit(quantized_train_features, train_labels)
    probs = clf.predict_proba(quantized_test_features)

    type_output_dict = {'probs': [], 'labels': []}
    results_by_source_type = {}
    results_by_source_type['overall'] = copy.deepcopy(type_output_dict)
    results_by_source_type['overall']['probs'] = probs
    results_by_source_type['overall']['labels'] = test_labels
    results_by_source_type['overall']['acc'] = np.mean(np.argmax(probs, 1) == test_labels)
    precision, recall, thresholds = precision_recall_curve(test_labels, probs[:, 1])
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
            results_by_source_type[source_type]['acc']= np.mean(np.argmax(probs[source_samples_idx], 1) == test_labels[source_samples_idx])
            precision, recall, thresholds = precision_recall_curve(test_labels[source_samples_idx], probs[source_samples_idx, 1])
            results_by_source_type[source_type]['auc'] = auc(recall, precision)
            if len(precision) == 2:
                results_by_source_type[source_type]['auc'] = 0.5
    return results_by_source_type, clf


def predict(clf, features):
    quantized_features = np.argsort(np.argsort(features, axis=1), axis=1)
    probs = clf.predict_proba(quantized_features)
    return probs