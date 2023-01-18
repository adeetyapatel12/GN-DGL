import numpy as np
import torch
import torch.nn.functional as F
import random
import json
from torch.autograd import Variable

def div_loss_calc(preds, labels, temp, device):
    if preds[0] == None:
        return 0

    # all_probs = torch.softmax(preds/ temp, dim=2)
    all_probs = torch.softmax(torch.stack(preds).to(device)/ temp, dim=2)

    n_heads = all_probs.shape[0]

    if n_heads < 2:
        return 0
    batch_size = all_probs.shape[1]
    n_class = all_probs.shape[2]

    label_inds = torch.ones(batch_size, n_class).to(device)
    label_inds[range(batch_size), labels] = 0

    # removing the gt prob
    probs = all_probs * label_inds.unsqueeze(0)
    # re-normalize such that probs sum to 1
    probs /= (probs.sum(dim = 2, keepdim = True) + 1e-8)
    probs = probs / torch.sqrt(((probs ** 2).sum(dim = 2, keepdim = True) + 1e-8))  # l2 normed
    cov_mat = torch.einsum('kij,lij->ikl', probs, probs)
    pairwise_inds = 1 - torch.eye(n_heads).to(device)
    den = batch_size * (n_heads - 1) * n_heads
    loss = (cov_mat * pairwise_inds).sum() / den
    return loss

def get_ensemble_logits(all_logits, device='cpu', ensemble_type='layerwise'):
    local_module_num = len(all_logits)
    if ensemble_type == 'layerwise':
        ensemble_weight = 2 ** torch.arange(local_module_num, device=device) / \
                               sum(2 ** torch.arange(local_module_num, device=device))

    elif ensemble_type == 'last_layer_only':
        ensemble_weight = torch.zeros(local_module_num, device=device)
        ensemble_weight[-1] = 1
    else:
        raise NotImplementedError

    # print(ensemble_weight)

    weighted_logits = []
    weighted_thread_logits = []
    for local_module_i in range(local_module_num):
        ## Single thread logits
        softmax_thread_logits_i = torch.softmax(all_logits[str(local_module_i)][0].to(device), dim=1)
        weighted_thread_logits.append(ensemble_weight[local_module_i] * softmax_thread_logits_i.squeeze())

        ## All threads' logits combined
        softmax_logits_i = torch.softmax(torch.stack(all_logits[str(local_module_i)], dim=0).to(device), dim=2)
        weighted_logits.append(ensemble_weight[local_module_i] *
                                         torch.mean(softmax_logits_i, dim=0).squeeze())

    thread_logits = torch.mean(torch.stack(weighted_thread_logits, dim=0), dim=0).squeeze()
    logits = torch.mean(torch.stack(weighted_logits, dim=0), dim=0).squeeze()

    return thread_logits, logits

