import torch
import numpy as np
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from torch import Tensor
import copy
from tqdm import tqdm

class Trainer():
    def __init__(self, max_num_epochs, criteria, optimizer, eval_metric,  eval_interval=1,  device='cpu', verbose=3, early_stop=None):
        self.max_num_epochs = max_num_epochs
        self.criteria = criteria
        self.optimizer = optimizer
        self.eval_metric = eval_metric
        self.device = device
        self.eval_interval = eval_interval

    def train(self, train_loader, eval_loader, model):
        raise NotImplemented

    def train_single_epoch(self, model, train_loader):
        raise NotImplemented

    def eval(self, model, eval_loader):
        raise NotImplemented

    def test(self, model, test_loader, metric=None):
        raise NotImplemented


class ClassifierTrainer(Trainer):
    def __init__(self, max_num_epochs, criteria, intermediate_criteria, intermediate_loss_weight,  optimizer,
                 eval_metric, eval_interval, device, verbose=3, early_stop=None):
        super().__init__(max_num_epochs, criteria, optimizer, eval_metric, eval_interval, device, verbose, early_stop)
        self.intermediate_criteria = intermediate_criteria
        if intermediate_loss_weight:
            self.intermediate_loss_weight = torch.tensor(intermediate_loss_weight, requires_grad=False).to(self.device)
        else:
            self.intermediate_loss_weight = None

    def train(self, train_loader, eval_loader, model, max_evals_no_improvement=8):

        best_model = None
        self.losses = {'train': [], 'validation': []}
        self.metrics = {'train': [], 'validation': []}
        best_loss, best_auc, best_acc, best_epoch = 0,0,0,0
        best_eval_loss = np.inf
        no_improvement_counter = 0
        n_samples_per_epoch = len(train_loader.dataset)

        for epoch in range(self.max_num_epochs):
            epoch_loss, epoch_intermediate_loss, epoch_classifier_loss, epoch_hits =\
                self.train_single_epoch(model,train_loader)

            self.losses['train'].append(epoch_loss/(n_samples_per_epoch))
            self.metrics['train'].append(epoch_hits/n_samples_per_epoch)
            print('epoch {}, train_loss: {:.2e}, classifier_loss: {:.2e}, intermediate_loss:{:.2e} train_acc:{:.2f}'
                  .format(epoch, self.losses['train'][-1], epoch_classifier_loss/n_samples_per_epoch,
                          epoch_intermediate_loss/n_samples_per_epoch,  epoch_hits/n_samples_per_epoch))

            if (np.mod(epoch, self.eval_interval) == 0 and epoch) or (epoch+1 == self.max_num_epochs):
                avg_eval_loss, avg_eval_intermediate_loss, avg_eval_classifier_loss, eval_acc, eval_auc, _, _ = \
                    self.eval(model, eval_loader)

                self.losses['validation'].append(avg_eval_loss)
                self.metrics['validation'].append(eval_acc)
                print('epoch {}, validation loss:{:.2e}, validation classifier loss:'
                      ' {:.2e}, validation intermediate loss: {:.2e}, validation acc: {:.2f}, validation auc: {:.3f}'
                      .format(epoch, avg_eval_loss, avg_eval_classifier_loss, avg_eval_intermediate_loss,
                              eval_acc, eval_auc))

                if best_auc < eval_auc:
                    best_eval_loss = avg_eval_loss
                    no_improvement_counter = 0
                    best_loss, best_auc, best_acc, best_epoch, _, _ = avg_eval_loss, eval_auc, eval_acc, epoch, _, _
                    best_model = copy.deepcopy(model)
                else:
                    no_improvement_counter += 1

                if no_improvement_counter == max_evals_no_improvement:
                    print('early stopping on epoch {}, best epoch {}'.format(epoch, best_epoch))
                    break
                if epoch + 1 == self.max_num_epochs:
                    print('reached maximum number of epochs, best epoch {}'.format(best_epoch))

        train_stats = {'best_val_loss':best_eval_loss, 'best_acc':best_acc, 'best_auc':best_auc,
                       'best_epoch':best_epoch}
        return train_stats, best_model

    def train_single_epoch(self, model, train_loader):
        epoch_loss = 0
        epoch_intermediate_loss = 0
        epoch_classifier_loss = 0
        hits = 0
        model.train()
        for source_batch, terminal_batch, labels_batch, pair_batch, pair_source_type, pair_degree in train_loader:
            if self.device != 'cpu':
                source_batch, terminal_batch, labels_batch, pair_degree =\
                    (x.to(self.device) for x in[source_batch, terminal_batch, labels_batch, pair_degree])
            if torch.get_default_dtype() is torch.float32:
                source_batch, terminal_batch, pair_degree = source_batch.float(), terminal_batch.float(), pair_degree.float()
            out, pred, pre_pred = model(source_batch, terminal_batch, pair_degree)
            classifier_loss = self.criteria(out, labels_batch)
            if self.intermediate_loss_weight:
                intermediate_loss = self.intermediate_criteria(torch.squeeze(pre_pred),
                                        torch.repeat_interleave(labels_batch, model.n_experiments).to(torch.get_default_dtype()))/model.n_experiments
                epoch_intermediate_loss += intermediate_loss.item()
                loss = ((1-self.intermediate_loss_weight) * classifier_loss) +\
                       (self.intermediate_loss_weight * intermediate_loss)
            else:
                loss = classifier_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            epoch_classifier_loss += classifier_loss.item()
            hits += torch.sum(pred == labels_batch).item()

        return epoch_loss, epoch_intermediate_loss, epoch_classifier_loss, hits

    def eval(self, model, eval_loader, output_probs= False):
        n_samples = len(eval_loader.dataset)
        eval_loss = 0
        eval_epoch_intermediate_eval_loss = 0
        epoch_classifier_loss = 0
        hits = 0

        all_outs = []
        all_labels = []
        model.eval()
        with torch.no_grad():
            for eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_batch, \
                eval_pair_source_type_batch, eval_pair_degree_batch in eval_loader:
                if self.device != 'cpu':
                    eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_degree_batch =\
                        (x.to(self.device) for x in[eval_source_batch, eval_terminal_batch, eval_labels_batch,
                                                    eval_pair_degree_batch])
                if torch.get_default_dtype() is torch.float32:
                    eval_source_batch, eval_terminal_batch, eval_pair_degree_batch =\
                        eval_source_batch.float(), eval_terminal_batch.float(), eval_pair_degree_batch.float()

                out, pred, pre_pred = model(eval_source_batch, eval_terminal_batch, eval_pair_degree_batch)

                classifier_loss = self.criteria(out, eval_labels_batch)
                if self.intermediate_loss_weight:
                    eval_intermediate_loss = self.intermediate_criteria(torch.squeeze(pre_pred),
                                            torch.repeat_interleave(eval_labels_batch, model.n_experiments).to(torch.get_default_dtype()))/model.n_experiments
                    eval_epoch_intermediate_eval_loss += eval_intermediate_loss.item()

                    loss = ((1-self.intermediate_loss_weight) * classifier_loss) +\
                           (self.intermediate_loss_weight * eval_intermediate_loss)
                else:
                    loss = classifier_loss

                eval_loss += loss.item()
                epoch_classifier_loss += classifier_loss.item()
                hits += torch.sum(pred == eval_labels_batch).item()
                all_outs.append(out.detach().cpu().numpy())
                all_labels.append(eval_labels_batch)

            # probs = torch.nn.functional.softmax(torch.squeeze(torch.cat(all_outs, 0)), dim=1).cpu().detach().numpy()
            probs = torch.nn.functional.softmax(torch.squeeze(Tensor(np.concatenate(all_outs, 0))), dim=1)
            all_labels = torch.squeeze(torch.cat(all_labels)).cpu().detach().numpy()

        if output_probs:
            return probs, all_labels
        else:
            precision, recall, thresholds = precision_recall_curve(all_labels, probs[:, 1])
            mean_auc = auc(recall, precision)

            avg_eval_loss = eval_loss / n_samples
            avg_eval_intermediate_loss = eval_epoch_intermediate_eval_loss/n_samples
            avg_eval_classifier_loss = epoch_classifier_loss / n_samples
            eval_acc = hits / n_samples
            return avg_eval_loss, avg_eval_intermediate_loss, avg_eval_classifier_loss, eval_acc, mean_auc, precision, recall

    def predict(self, model, eval_loader):
        all_outs = []
        all_pairs = []
        n_batches = int(len(eval_loader.dataset) // eval_loader.batch_size + 1)
        model.eval()
        with torch.no_grad():
            for eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_batch,\
                eval_pair_source_type_batch, eval_pair_degree_batch in\
                    tqdm(eval_loader, desc='Predicting edges direction' ,total=n_batches):
                if self.device != 'cpu':
                    eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_degree_batch =\
                        (x.to(self.device) for x in[eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_degree_batch])
                if torch.get_default_dtype() is torch.float32:
                    eval_source_batch, eval_terminal_batch, eval_pair_degree_batch =\
                        eval_source_batch.float(), eval_terminal_batch.float(), eval_pair_degree_batch.float()
                out, pred, pre_pred = model(eval_source_batch, eval_terminal_batch, eval_pair_degree_batch)

                all_outs.append(out.detach().cpu().numpy())
                all_pairs.append(torch.stack(eval_pair_batch).detach().cpu().numpy())

            all_probs = torch.nn.functional.softmax(torch.squeeze(Tensor(np.concatenate(all_outs, 0))), dim=1).numpy()
            all_pairs = np.hstack(all_pairs).T
            all_pairs = np.array([[eval_loader.dataset.col_idx_to_id[x] for x in pair] for pair in all_pairs])
        return {'probs':all_probs, 'pairs':all_pairs}

    def eval_by_source(self, model, eval_loader, output_probs= False, by_source_type=False):
        type_output_dict = {'probs': [], 'labels': [], 'intermediate_loss': 0, 'classifier_loss': 0, 'loss': 0, 'hits': 0}
        type_result_dict = {'mean_auc': 0, 'avg_loss': 0, 'avg_classifier_loss': 0, 'avg_intermediate_loss': 0, 'acc':0}
        output_per_type_dict = dict()
        output_per_type_dict['overall'] = copy.deepcopy(type_output_dict)
        results_per_type_dict = dict()
        results_per_type_dict['overall'] = copy.deepcopy(type_result_dict)

        model.eval()
        with torch.no_grad():
            for eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_batch,\
                eval_pair_source_type_batch, eval_pair_degree_batch in eval_loader:
                if self.device != 'cpu':
                    eval_source_batch, eval_terminal_batch, eval_labels_batch, eval_pair_degree_batch =\
                        (x.to(self.device) for x in[eval_source_batch, eval_terminal_batch, eval_labels_batch,
                                                    eval_pair_degree_batch])
                if torch.get_default_dtype() is torch.float32:
                    eval_source_batch, eval_terminal_batch, eval_pair_degree_batch =\
                        eval_source_batch.float(), eval_terminal_batch.float(), eval_pair_degree_batch.float()

                unique_types = np.unique(eval_pair_source_type_batch)
                for unique_type in unique_types:
                    type_idx = [x for x, xx in enumerate(eval_pair_source_type_batch) if xx == unique_type]
                    type_source_batch = eval_source_batch[type_idx]
                    type_terminal_batch = eval_terminal_batch[type_idx]
                    type_labels = eval_labels_batch[type_idx]
                    type_pair_degree_batch = eval_pair_degree_batch[type_idx]
                    out, pred, pre_pred = model(type_source_batch, type_terminal_batch, type_pair_degree_batch)

                    classifier_loss = self.criteria(out, type_labels)
                    if self.intermediate_loss_weight:
                        eval_intermediate_loss = self.intermediate_criteria(torch.squeeze(pre_pred, dim=1),
                                                torch.repeat_interleave(type_labels, model.n_experiments).to(torch.get_default_dtype())) / model.n_experiments

                        loss = ((1-self.intermediate_loss_weight) * classifier_loss) +\
                               (self.intermediate_loss_weight * eval_intermediate_loss)
                    else:
                        loss = classifier_loss

                    if unique_type not in output_per_type_dict:
                        output_per_type_dict[unique_type] = copy.deepcopy(type_output_dict)

                    if self.intermediate_loss_weight:
                        output_per_type_dict[unique_type]['intermediate_loss'] += eval_intermediate_loss.item()
                        output_per_type_dict['overall']['intermediate_loss'] += eval_intermediate_loss.item()
                    else:
                        output_per_type_dict[unique_type]['intermediate_loss'] += 0
                        output_per_type_dict['overall']['intermediate_loss'] += 0
                    output_per_type_dict[unique_type]['classifier_loss'] += classifier_loss.item()
                    output_per_type_dict[unique_type]['loss'] += loss.item()
                    output_per_type_dict[unique_type]['hits'] += torch.sum(pred == type_labels).item()
                    output_per_type_dict[unique_type]['probs'].append(out.cpu().detach().numpy())
                    output_per_type_dict[unique_type]['labels'].append(type_labels)

                    output_per_type_dict['overall']['classifier_loss'] += classifier_loss.item()
                    output_per_type_dict['overall']['loss'] += loss.item()
                    output_per_type_dict['overall']['hits'] += torch.sum(pred == type_labels).item()
                    output_per_type_dict['overall']['probs'].append(out.cpu().detach().numpy())
                    output_per_type_dict['overall']['labels'].append(type_labels)

            for unique_type in output_per_type_dict.keys():
                output_per_type_dict[unique_type]['probs'] =\
                    torch.nn.functional.softmax(torch.squeeze(Tensor(np.concatenate(output_per_type_dict[unique_type]['probs'], 0))), dim=1)
                output_per_type_dict[unique_type]['labels'] =\
                    torch.squeeze(torch.cat(output_per_type_dict[unique_type]['labels'])).cpu().detach().numpy()

        if output_probs:
            return output_per_type_dict

        else:
            for unique_type in output_per_type_dict.keys():
                results_per_type_dict[unique_type] = copy.deepcopy(type_result_dict)
                n_type_samples = len(output_per_type_dict[unique_type]['labels'])
                precision, recall, thresholds = precision_recall_curve(output_per_type_dict[unique_type]['labels'],
                                                                       output_per_type_dict[unique_type]['probs'][:, 1])
                results_per_type_dict[unique_type]['mean_auc'] = auc(recall, precision)
                results_per_type_dict[unique_type]['avg_loss'] = output_per_type_dict[unique_type]['loss'] / n_type_samples
                results_per_type_dict[unique_type]['avg_intermediate_loss']= output_per_type_dict[unique_type]['intermediate_loss'] / n_type_samples
                results_per_type_dict[unique_type]['avg_classifier_loss'] = output_per_type_dict[unique_type]['classifier_loss'] / n_type_samples
                results_per_type_dict[unique_type]['acc'] = output_per_type_dict[unique_type]['hits'] / n_type_samples

            return results_per_type_dict
