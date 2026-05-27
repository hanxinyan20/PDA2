from model.fc_net_mcdo import MCDropoutFullyConnectedNet
from dataset import SampleWiseData
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .trainer_utils import EarlyStopping
from copy import deepcopy
import math
import json
from prettytable import PrettyTable
import pandas as pd
import torch
from torch import Tensor
from torch.nn.functional import log_softmax, nll_loss
import math
from typing import Tuple,Callable, Sequence, Union, List
from torch.utils.data import DataLoader
from copy import deepcopy
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.distributions import Gumbel
from dataclasses import dataclass
from .joint_entropy import *
def logmeanexp(x: Tensor, dim: int, keepdim: bool = False) -> Tensor:
    return torch.logsumexp(x, dim=dim, keepdim=keepdim) - math.log(x.shape[dim])
def accuracy_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float], [1,]
    """
    return count_correct_from_conditionals(predictions, labels) / len(predictions)  # [K,]
def count_correct_from_conditionals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, K, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == labels[:, None]  # [N, K]
    return torch.sum(is_correct, dim=0)  #  [K,]
def count_correct_from_marginals(predictions: Tensor, labels: Tensor) -> Tensor:
    """
    Arguments:
        predictions: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[int], [1,]
    """
    is_correct = torch.argmax(predictions, dim=-1) == labels  # [N,]
    return torch.sum(is_correct)  #  [1,]
def nll_loss_from_probs(probs: Tensor, labels: Tensor, reduction: str) -> Tensor:
    """
    Arguments:
        probs: Tensor[float], [N, Cl]
        labels: Tensor[int], [N,]

    Returns:
        Tensor[float]
    """
    probs = torch.clamp(probs, min=torch.finfo(probs.dtype).eps)  # [N, Cl]
    return nll_loss(torch.log(probs), labels, reduction=reduction)
def check(
    scores: Tensor,
    min_value: float = 0.0,
    max_value: float = math.inf,
    epsilon: float = 1e-6,
    score_type: str = "",
) -> Tensor:
    """
    Warn if any element of scores is negative, a NaN or exceeds max_value.

    We set epilson = 1e-6 based on the fact that torch.finfo(torch.float).eps ~= 1e-7.
    """
    if not torch.all((scores >= min_value - epsilon) & (scores <= max_value + epsilon)):
        min_score = torch.min(scores).item()
        max_score = torch.max(scores).item()
        print(f"Invalid {score_type} score (min = {min_score}, max = {max_score})")

    return scores
def entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    If p(y_i|x) is 0, we make sure p(y_i|x) log p(y_i|x) evaluates to 0, not NaN.

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.xlogy(probs, probs), dim=-1)
def marginal_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    probs = torch.mean(probs, dim=1)  # [N, Cl]

    scores = entropy_from_probs(probs)  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]

def conditional_epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p, N_t]
    """
    # Estimate the joint predictive distribution.
    probs_pool = probs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    probs_targ = probs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    probs_joint = probs_pool * probs_targ  # [N_p, N_t, K, Cl, Cl]
    probs_joint = torch.mean(probs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the marginal predictive distributions.
    probs_pool = torch.mean(probs_pool, dim=2)  # [N_p, 1, Cl, 1]
    probs_targ = torch.mean(probs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the product of the marginal predictive distributions.
    probs_pool_targ_indep = probs_pool * probs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    nonzero_joint = probs_joint > 0  # [N_p, N_t, Cl, Cl]
    log_term = torch.clone(probs_joint)  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] = torch.log(probs_joint[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    log_term[nonzero_joint] -= torch.log(probs_pool_targ_indep[nonzero_joint])  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(probs_joint * log_term, dim=(-2, -1))  # [N_p, N_t]

    return scores  # [N_p, N_t]
def epig_from_probs(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    _, _, Cl = probs_pool.shape

    scores = conditional_epig_from_probs(probs_pool, probs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]
def epig_from_probs_using_matmul(probs_pool: Tensor, probs_targ: Tensor) -> Tensor:
    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    N_t, K, Cl = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ = marginal_entropy_from_probs(probs_targ)  # [N_t,]

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * Cl)  # [K, N_t * Cl]
    probs_joint = probs_pool @ probs_targ / K  # [N_p, Cl, N_t * Cl]

    entropy_joint = -torch.sum(torch.xlogy(probs_joint, probs_joint), dim=(-2, -1)) / N_t  # [N_p,]

    scores = entropy_pool + torch.mean(entropy_targ) - entropy_joint  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]
def epig_from_logprobs_using_matmul(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    probs_pool = torch.exp(logprobs_pool)  # [N_p, K, Cl]
    probs_targ = torch.exp(logprobs_targ)  # [N_t, K, Cl]

    return epig_from_probs_using_matmul(probs_pool, probs_targ)  # [N_p,]
def epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert logprobs_pool.ndim == logprobs_targ.ndim == 3

    _, _, Cl = logprobs_pool.shape

    scores = conditional_epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p, N_t]
    scores = torch.mean(scores, dim=-1)  # [N_p,]
    scores = check(scores, max_value=math.log(Cl**2), score_type="EPIG")  # [N_p,]

    return scores  # [N_p,]
def conditional_epig_from_logprobs(logprobs_pool: Tensor, logprobs_targ: Tensor) -> Tensor:
    """
    EPIG(x|x_*) = I(y;y_*|x,x_*)
                = KL[p(y,y_*|x,x_*) || p(y|x)p(y_*|x_*)]
                = ∑_{y} ∑_{y_*} p(y,y_*|x,x_*) log(p(y,y_*|x,x_*) / p(y|x)p(y_*|x_*))

    Arguments:
        logprobs_pool: Tensor[float], [N_p, K, Cl]
        logprobs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    # Estimate the log of the joint predictive distribution.
    logprobs_pool = logprobs_pool[:, None, :, :, None]  # [N_p, 1, K, Cl, 1]
    logprobs_targ = logprobs_targ[None, :, :, None, :]  # [1, N_t, K, 1, Cl]
    logprobs_joint = logprobs_pool + logprobs_targ  # [N_p, N_t, K, Cl, Cl]
    logprobs_joint = logmeanexp(logprobs_joint, dim=2)  # [N_p, N_t, Cl, Cl]

    # Estimate the log of the marginal predictive distributions.
    logprobs_pool = logmeanexp(logprobs_pool, dim=2)  # [N_p, 1, Cl, 1]
    logprobs_targ = logmeanexp(logprobs_targ, dim=2)  # [1, N_t, 1, Cl]

    # Estimate the log of the product of the marginal predictive distributions.
    logprobs_joint_indep = logprobs_pool + logprobs_targ  # [N_p, N_t, Cl, Cl]

    # Estimate the conditional expected predictive information gain for each pair of examples.
    # This is the KL divergence between probs_joint and probs_joint_indep.
    log_term = logprobs_joint - logprobs_joint_indep  # [N_p, N_t, Cl, Cl]
    scores = torch.sum(torch.exp(logprobs_joint) * log_term, dim=(-2, -1))  # [N_p, N_t]

    return scores  # [N_p, N_t]
def get_next(dataloader: DataLoader) -> Union[Tensor, Tuple]:
    try:
        return next(dataloader)
    except:
        dataloader = iter(dataloader)
        return next(dataloader)
class Dictionary(dict):
    def append(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += [dictionary[key]]
            else:
                self[key] = [dictionary[key]]

    def extend(self, dictionary: dict) -> None:
        for key in dictionary:
            if key in self:
                self[key] += dictionary[key]
            else:
                self[key] = dictionary[key]

    def concatenate(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            if isinstance(dictionary[key][0], np.ndarray):
                dictionary[key] = np.concatenate(dictionary[key])
            elif isinstance(dictionary[key][0], Tensor):
                if dictionary[key][0].ndim == 0:
                    dictionary[key] = torch.tensor(dictionary[key])
                else:
                    dictionary[key] = torch.cat(dictionary[key])
            else:
                raise TypeError

        return dictionary

    def numpy(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key].numpy()

        return dictionary

    def torch(self) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = torch.tensor(dictionary[key])

        return dictionary

    def subset(self, inds: Sequence) -> dict:
        dictionary = deepcopy(self)

        for key in dictionary:
            dictionary[key] = dictionary[key][inds]

        return dictionary

    def save_to_csv(self, filepath: Path, formatting: Union[Callable, dict] = None) -> None:
        table = pd.DataFrame(self)

        if callable(formatting):
            table = table.applymap(formatting)

        elif isinstance(formatting, dict):
            for key in formatting:
                if key in self:
                    table[key] = table[key].apply(formatting[key])

        table.to_csv(filepath, index=False)

    def save_to_npz(self, filepath: Path) -> None:
        np.savez(filepath, **self)
def prepend_to_keys(dictionary: dict, string: str) -> dict:
    return {f"{string}_{key}": value for key, value in dictionary.items()}


def entropy_from_probs(probs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    If p(y_i|x) is 0, we make sure p(y_i|x) log p(y_i|x) evaluates to 0, not NaN.

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.xlogy(probs, probs), dim=-1)

@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def conditional_entropy_from_probs(probs: Tensor) -> Tensor:
    """
    E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    scores = entropy_from_probs(probs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="CE")  # [N,]

    return scores  # [N,]


def bald_from_probs(probs: Tensor) -> Tensor:
    """
    BALD(x) = E_{p(θ)}[H[p(y|x)] - H[p(y|x,θ)]]
            = H[p(y|x)] - E_{p(θ)}[H[p(y|x,θ)]]
            = H[E_{p(θ)}[p(y|x,θ)]] - E_{p(θ)}[H[p(y|x,θ)]]

    BALD(x) = 0 for a deterministic model but numerical instabilities can lead to nonzero scores.

    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/01_batchbald.ipynb

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    print("probs:", probs.shape)
    marg_entropy = marginal_entropy_from_probs(probs)  # [N,]
    cond_entropy = conditional_entropy_from_probs(probs)  # [N,]

    scores = marg_entropy - cond_entropy  # [N,]
    scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="BALD")  # [N,]

    return scores  # [N,]
def entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    Arguments:
        logprobs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.exp(logprobs) * logprobs, dim=-1)


def marginal_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim == 3

    logprobs = logmeanexp(logprobs, dim=1)  # [N, Cl]

    scores = entropy_from_logprobs(logprobs)  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]


def conditional_entropy_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    E_{p(θ)}[H[p(y|x,θ)]]

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert logprobs.ndim == 3

    scores = entropy_from_logprobs(logprobs)  # [N, K]
    scores = torch.mean(scores, dim=-1)  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="CE")  # [N,]

    return scores  # [N,]


def bald_from_logprobs(logprobs: Tensor) -> Tensor:
    """
    BALD(x) = E_{p(θ)}[H[p(y|x)] - H[p(y|x,θ)]]
            = H[p(y|x)] - E_{p(θ)}[H[p(y|x,θ)]]
            = H[E_{p(θ)}[p(y|x,θ)]] - E_{p(θ)}[H[p(y|x,θ)]]

    BALD(x) = 0 for a deterministic model but numerical instabilities can lead to nonzero scores.

    References:
        https://github.com/BlackHC/batchbald_redux/blob/master/01_batchbald.ipynb

    Arguments:
        logprobs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    marg_entropy = marginal_entropy_from_logprobs(logprobs)  # [N,]
    cond_entropy = conditional_entropy_from_logprobs(logprobs)  # [N,]

    scores = marg_entropy - cond_entropy  # [N,]
    scores = check(scores, max_value=math.log(logprobs.shape[-1]), score_type="BALD")  # [N,]

    return scores  # [N,]
class ALTrainer:
    def __init__(self, acquisition_config, training_config, seed, model_config, sample_wise_data: SampleWiseData, device, logger):
        self.acquisition_config = acquisition_config
        self.training_config = training_config
        self.model_config = model_config
        self.sample_wise_data = sample_wise_data
        self.device = device
        self.logger = logger
        self.rng = np.random.default_rng(seed)
        self.n_samples_train = self.acquisition_config['acquisition_function']['n_samples_train']
        self.n_samples_test = self.acquisition_config['acquisition_function']['n_samples_test']
    
    
    def acquire_and_finetune(self):
        summary_loss_dict = {'valid_loss': [], 'test_loss': [], 'test_acc_from_0': []}
        test_loader = DataLoader(
            dataset=self.sample_wise_data.get_tgt_test_dataset(),
            batch_size=self.training_config['batch_size'],
            shuffle=False,
            drop_last=False,
        )
        torch.save(self.sample_wise_data.x_train, f"{self.acquisition_config['save_dir']}/x_train.pt")
        torch.save(self.sample_wise_data.y_train, f"{self.acquisition_config['save_dir']}/y_train.pt")
        torch.save(self.sample_wise_data.x_tgt_valid, f"{self.acquisition_config['save_dir']}/x_tgt_valid.pt")
        torch.save(self.sample_wise_data.y_tgt_valid, f"{self.acquisition_config['save_dir']}/y_tgt_valid.pt")
        torch.save(self.sample_wise_data.x_tgt_test, f"{self.acquisition_config['save_dir']}/x_tgt_test.pt")
        torch.save(self.sample_wise_data.y_tgt_test, f"{self.acquisition_config['save_dir']}/y_tgt_test.pt")
        torch.save(self.sample_wise_data.x_src_valid, f"{self.acquisition_config['save_dir']}/x_src_valid.pt")
        torch.save(self.sample_wise_data.y_src_valid, f"{self.acquisition_config['save_dir']}/y_src_valid.pt")
        for r in range(self.acquisition_config['n_rounds']):
            self.model = MCDropoutFullyConnectedNet(
                input_features=self.sample_wise_data.n_features,
                hidden_sizes=self.model_config['layer_dims'],
                output_size=self.sample_wise_data.n_classes,
                activation_fn=nn.ReLU,
                dropout_rate=self.model_config['dropout_rate'],
                use_input_dropout=self.model_config['use_input_dropout']
            )
            if self.training_config['optimizer'] == 'adam':
                self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.training_config['learning_rate'],
                    weight_decay=self.training_config['weight_decay']
                )
            elif self.training_config['optimizer'] == 'sgd':
                self.optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.training_config['learning_rate'],
                    weight_decay=self.training_config['weight_decay']
                )
            else:
                raise ValueError(f"Unsupported optimizer: {self.training_config['optimizer']}")
            summary_loss_dict['valid_loss'].append(self.train(train_dataset=self.sample_wise_data.get_train_dataset(),valid_dataset=self.sample_wise_data.get_tgt_valid_dataset(),early_stop_config=self.training_config['early_stop'],max_steps=self.training_config['max_steps']+int(math.sqrt(100*r)),batch_size=self.training_config['batch_size']))
            with torch.inference_mode():
                test_metrics = self.test(test_loader, self.sample_wise_data.n_classes)
                summary_loss_dict['test_acc_from_0'].append(test_metrics['acc'])
                summary_loss_dict['test_loss'].append(test_metrics['nll'])
            pool_dataloader = DataLoader(
                dataset=self.sample_wise_data.get_pool_dataset(),
                batch_size=self.training_config['batch_size'],
                shuffle=False,
                drop_last=False,
            )
            target_inputs = self.sample_wise_data.get_tgt_valid_dataset().tensors[0]
            acq_kwargs = dict(
                inputs_targ = target_inputs,
                loader = pool_dataloader,
                method = self.acquisition_config['acquisition_function']['method'],
                seed = self.rng.choice(int(1e6)),
            )
            with torch.inference_mode():
                scores = self.estimate_uncertainty(**acq_kwargs)
            scores = torch.log(scores) + Gumbel(loc=0, scale=1).sample(scores.shape)
            acquired_pool_inds = torch.argsort(scores)[-self.acquisition_config['n_samples_per_round']:]
            acquired_pool_inds = acquired_pool_inds.tolist()
            self.sample_wise_data.remove_from_pool_to_train(acquired_pool_inds)
            acquire_table = PrettyTable()
            acquire_table.field_names = ['Round', 'test_acc', 'test_loss']+self.sample_wise_data.other_regions + ['valid_loss_record']
            acquire_table.add_row([r, summary_loss_dict['test_acc_from_0'][-1], summary_loss_dict['test_loss'][-1]]+self.sample_wise_data.region_acquired_times+[summary_loss_dict['valid_loss'][-1]])
            self.logger.info("%s", acquire_table)
            if r % self.acquisition_config['save_rounds'] == self.acquisition_config['save_rounds']-1:
                x_acquired, y_acquired = self.sample_wise_data.get_acquired_samples()
                torch.save(x_acquired, f"{self.acquisition_config['save_dir']}/x_acquired_{r}.pt")
                torch.save(y_acquired, f"{self.acquisition_config['save_dir']}/y_acquired_{r}.pt")
                
        with torch.inference_mode():
            test_metrics = self.test(test_loader, self.sample_wise_data.n_classes)
            summary_loss_dict['test_acc_from_0'].append(test_metrics['acc'])
            summary_loss_dict['test_loss'].append(test_metrics['nll'])
            
        self.logger.info("%s", json.dumps(summary_loss_dict, indent=4))
        return summary_loss_dict['test_acc_from_0']

    def train(self, train_dataset, valid_dataset, early_stop_config, max_steps, batch_size):
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        valid_loss_record = []
        if early_stop_config['enabled']:
            early_stopping = EarlyStopping(
                patience=early_stop_config['patience'],
                delta=early_stop_config['delta'],
                verbose=True,
            )
        for step in range(max_steps):
            train_metrics = self.train_step(train_loader)

            if step % 15 == 0:
                with torch.inference_mode():
                    val_metrics = self.test(val_loader)
                valid_loss_record.append(val_metrics['nll'])
                
                if early_stop_config['enabled']:
                    early_stopping(val_metrics['nll'], self.model)
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
        return valid_loss_record
                    
    
    def eval_mode(self) -> None:
        self.model.eval()

    def conditional_predict(
        self, inputs: Tensor, n_model_samples: int, independent: bool
    ) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int
            independent: bool

        Returns:
            Tensor[float], [N, K, Cl]
        """
        features = self.model(inputs, n_model_samples)  # [N, K, Cl]
        return log_softmax(features, dim=-1)  # [N, K, Cl]
    def marginal_predict(self, inputs: Tensor, n_model_samples: int) -> Tensor:
        """
        Arguments:
            inputs: Tensor[float], [N, *F]
            n_model_samples: int

        Returns:
            Tensor[float], [N, Cl]
        """
        logprobs = self.conditional_predict(inputs, n_model_samples, independent=True)  # [N, K, Cl]

        if n_model_samples == 1:
            return torch.squeeze(logprobs, dim=1)  # [N, Cl]
        else:
            return logmeanexp(logprobs, dim=1)  # [N, Cl]
    def evaluate_train(self, inputs: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]:
        """
        loss = 1/KN ∑_{j=1}^K ∑_{i=1}^N L(x_i,y_i,θ_j) where θ_j ~ p(θ)

        Here we use
            L_1(x_i,y_i,θ_j) = nll_loss(x_i,y_i,θ_j) = -log p(y_i|x_i,θ_j)
            L_2(x_i,y_i,θ_j) = binary_loss(x_i,y_i,θ_j) = {argmax p(y|x_i,θ_j) != y_i}
        """
        logprobs = self.conditional_predict(
            inputs, self.n_samples_train, independent=False
        )  # [N, K, Cl]

        acc = accuracy_from_conditionals(logprobs, labels)  # [K,]
        acc = torch.mean(acc)  # [1,]

        nll_loss_list = []
        for k in range(logprobs.shape[1]):
            nll_loss_list.append(nll_loss(logprobs[:, k], labels, reduction="mean"))
        nll = torch.mean(torch.stack(nll_loss_list))  # [1,]

        return acc, nll
    def evaluate_test(self, inputs: Tensor, labels: Tensor, n_classes: int = None) -> dict:
        logprobs = self.marginal_predict(inputs, self.n_samples_test)  # [N, Cl]

        if (n_classes is not None) and (n_classes < logprobs.shape[-1]):
            logprobs = logprobs[:, :n_classes]  # [N, n_classes]
            logprobs -= torch.logsumexp(logprobs, dim=-1, keepdim=True)  # [N, n_classes]
        n_correct = count_correct_from_marginals(logprobs, labels)  # [1,]
        nll = nll_loss(logprobs, labels, reduction="sum")  # [1,]

        return {"n_correct": n_correct, "nll": nll}
    def test(self, loader: DataLoader, n_classes: int = None) -> dict:
        """
        loss = 1/N ∑_{i=1}^N L(x_i,y_i)

        Here we use
            L_1(x_i,y_i) = bin_loss(x_i,y_i) = 1[argmax p(y|x_i) == y_i]
            L_2(x_i,y_i) = mae_loss(x_i,y_i) = |E[p(y|x_i)] - y_i|
            L_3(x_i,y_i) = mse_loss(x_i,y_i) = (E[p(y|x_i)] - y_i)^2
            L_4(x_i,y_i) = nll_loss(x_i,y_i) = -log p(y_i|x_i).

        For stochastic models we use
            p(y|x_i)  = E_{p(θ)}[p(y|x_i,θ)]
                     ~= 1/K ∑_{j=1}^K p(y|x_i,θ_j), θ_j ~ p(θ).
        """
        self.eval_mode()

        test_log = Dictionary()
        n_examples = 0

        for inputs, labels in loader:
            if n_classes is not None:
                test_log_update = self.evaluate_test(inputs, labels, n_classes)
            else:
                test_log_update = self.evaluate_test(inputs, labels)

            test_log.append(test_log_update)

            n_examples += len(inputs)
        
        test_log = test_log.concatenate()

        for metric, scores in test_log.items():
            test_log[metric] = torch.sum(scores).item() / n_examples

        if "n_correct" in test_log:
            test_log["acc"] = test_log.pop("n_correct")
        else:
            print("n_correct not in test_log")
        return test_log
    def estimate_uncertainty(
        self, loader: DataLoader, method: str, seed: int, inputs_targ: Tensor = None
    ) -> Tensor:
        self.eval_mode()
        estimate_epig_using_pool = False
        if estimate_epig_using_pool:
            scores = self.estimate_epig_using_pool(loader, n_input_samples=len(inputs_targ))  # [N,]

        else:
            scores = []

            for inputs_i, _ in loader:

                if method == "epig":
                    scores_i = self.estimate_epig_batch(inputs_i, inputs_targ)  # [B,]
                elif method == "bald":
                    scores_i = self.estimate_uncertainty_batch(inputs_i, method)  # [B,]
                else:
                    raise NotImplementedError
                scores += [scores_i.cpu()]

            scores = torch.cat(scores)  # [N,]
            print(len(scores))

        return scores  # [N,]
    def estimate_epig_batch(self, inputs_pool: Tensor, inputs_targ: Tensor) -> Tensor:
            logprobs = self.conditional_predict(
                torch.cat((inputs_pool, inputs_targ)), self.n_samples_test, independent=False
            )  # [N_p + N_t, K, Cl]

            logprobs_pool = logprobs[: len(inputs_pool)]  # [N_p, K, Cl]
            logprobs_targ = logprobs[len(inputs_pool) :]  # [N_t, K, Cl]

            self.use_matmul = True
            
            if self.use_matmul:
                scores = epig_from_logprobs_using_matmul(logprobs_pool, logprobs_targ)  # [N_p,]
            else:
                scores = epig_from_logprobs(logprobs_pool, logprobs_targ)  # [N_p,]

            return scores  # [N_p,]
    def estimate_epig_using_pool(self, loader: DataLoader, n_input_samples: int = None) -> Tensor:
        logprobs_cond = []

        for inputs, _ in loader:
            logprobs_cond_i = self.conditional_predict(
                inputs, self.n_samples_test, independent=True
            )  # [B, K, Cl]
            logprobs_cond += [logprobs_cond_i]

        logprobs_cond = torch.cat(logprobs_cond)  # [N, K, Cl]
        logprobs_marg = logmeanexp(logprobs_cond, dim=1)  # [N, Cl]
        logprobs_marg_marg = logmeanexp(logprobs_marg, dim=0, keepdim=True)  # [1, Cl]

        # Compute the weights, w(x_*) ~= ∑_{y_*} p_*(y_*) p_{pool}(y_*|x_*) / p_{pool}(y_*).
        target_class_dist = self.epig_cfg.target_class_dist
        target_class_dist = torch.tensor([target_class_dist], device=inputs.device)  # [1, Cl]
        target_class_dist /= torch.sum(target_class_dist)  # [1, Cl]
        log_ratio = logprobs_marg - logprobs_marg_marg  # [N, Cl]
        weights = torch.sum(target_class_dist * torch.exp(log_ratio), dim=-1)  # [N,]

        # Ensure that ∑_{x_*} w(x_*) == N.
        assert math.isclose(torch.sum(weights).item(), len(weights), rel_tol=1e-3)

        # Compute the weighted EPIG scores.
        scores = []

        if n_input_samples is not None:
            # We do not need to normalize the weights before passing them to torch.multinomial().
            inds = torch.multinomial(
                weights, num_samples=n_input_samples, replacement=True
            )  # [N_s,]

            logprobs_targ = logprobs_cond[inds]  # [N_s, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                if self.epig_cfg.use_matmul:
                    scores_i = epig_from_logprobs_using_matmul(
                        logprobs_cond_i, logprobs_targ
                    )  # [B,]
                else:
                    scores_i = epig_from_logprobs(logprobs_cond_i, logprobs_targ)  # [B,]

                scores += [scores_i.cpu()]

        else:
            logprobs_targ = logprobs_cond  # [N, K, Cl]

            for logprobs_cond_i in torch.split(logprobs_cond, len(inputs)):
                scores_i = epig_from_logprobs_using_weights(
                    logprobs_cond_i, logprobs_targ, weights
                )  # [B,]
                scores += [scores_i.cpu()]

        return torch.cat(scores)  # [N,]
    def train_step(self, loader: DataLoader) -> dict:
        inputs, labels = get_next(loader)  # [N, ...], [N,]

        self.model.train()

        acc, nll = self.evaluate_train(inputs, labels)  # [1,], [1,]

        self.optimizer.zero_grad()
        nll.backward()
        self.optimizer.step()

        return {"acc": acc.item(), "nll": nll.item()}
    
    def estimate_uncertainty_batch(self, inputs: Tensor, method: str) -> Tensor:
        uncertainty_estimator = self.uncertainty_estimators[method]

        logprobs = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )  # [N, K, Cl]

        return uncertainty_estimator(logprobs)  # [N,]      
    def get_batchbald_batch(
            self, inputs: Tensor, batch_size: int, dtype=None, device=None
        ) -> CandidateBatch:
        log_probs_N_K_C = self.conditional_predict(
            inputs, self.n_samples_test, independent=True
        )
        log_probs_N_K_C = log_probs_N_K_C.to(dtype=dtype, device=device)  # [N, K, Cl]
        N, K, C = log_probs_N_K_C.shape

        batch_size = min(batch_size, N)

        candidate_indices = []
        candidate_scores = []

        if batch_size == 0:
            return CandidateBatch(candidate_scores, candidate_indices)

        conditional_entropies_N = conditional_entropy_from_logprobs(log_probs_N_K_C)

        batch_joint_entropy = DynamicJointEntropy(
            self.n_samples_test, batch_size - 1, K, C, dtype=dtype, device=device
        )

        # We always keep these on the CPU.
        scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

        for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
            if i > 0:
                latest_index = candidate_indices[-1]
                batch_joint_entropy.add_variables(log_probs_N_K_C[latest_index : latest_index + 1])

            shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

            batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

            scores_N -= conditional_entropies_N + shared_conditinal_entropies
            scores_N[candidate_indices] = -float("inf")

            candidate_score, candidate_index = scores_N.max(dim=0)

            candidate_indices.append(candidate_index.item())
            candidate_scores.append(candidate_score.item())

        return CandidateBatch(candidate_scores, candidate_indices)        