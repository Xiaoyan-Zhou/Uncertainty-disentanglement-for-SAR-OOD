import numpy as np
import torch
import math
"""The dissonance of an opinion can happen when there is
an insufficient amount of evidence that can clearly support a
particular belief."""
# Calculate dissonance of a vector of alpha # paper:Multidimensional Uncertainty-Aware Evidential Neural Networks
def getDisn(alpha):
    evi = alpha - 1
    s = torch.sum(alpha, axis=1, keepdims=True)
    blf = evi / s
    idx = np.arange(alpha.shape[1])
    vac = evi.shape[1]/s # u=K/S, classnum=evi.shape[1]
    diss = 0
    Bal = lambda bi, bj: 1 - torch.abs(bi - bj) / (bi + bj + 1e-8)
    for i in idx:
        score_j_bal = [blf[:, j] * Bal(blf[:, j], blf[:, i]) for j in idx[idx != i]]
        score_j = [blf[:, j] for j in idx[idx != i]]
        diss += blf[:, i] * sum(score_j_bal) / (sum(score_j) + 1e-8)
    return diss, vac.squeeze()

### based on paper: Evidential uncertainty quantification: a variance-based perspective
def cal_uncertainty(alpha, UNCERTAINTY_mode, reduce=False):
    if UNCERTAINTY_mode == 'variance_class-wise':
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        _, index = torch.max(p, dim=1)
        variance = p - p ** 2
        EU = (alpha / S) * (1 - alpha / S) / (S + 1) # epistemic uncertainty
        AU = variance - EU # aleatoric uncertainty
        if reduce:
            AU = AU.sum() / alpha.shape[0]
            EU = EU.sum() / alpha.shape[0]
        return AU, EU
    elif UNCERTAINTY_mode in ['variance_sample-wise', 'variance']:
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        _, index = torch.max(p, dim=1)
        variance = p - p ** 2
        TU = variance.sum(dim=1)
        EU_class = (alpha / S) * (1 - alpha / S) / (S + 1) # epistemic uncertainty
        EU = EU_class.sum(dim=1)
        AU = TU - EU # aleatoric uncertainty
        if reduce:
            AU = AU.sum() / alpha.shape[0]
            EU = EU.sum() / alpha.shape[0]
        return AU, EU 
    elif UNCERTAINTY_mode == 'entropy':
        S = alpha.sum(dim=1, keepdim=True)
        p = alpha / S
        num_class = p.shape[1]
        _, index = torch.max(p, dim=1)
        row_indices = torch.arange(p.shape[0])

        entropy = - (p * (p + 1e-7).log()).sum(dim=1)
        Udata = ((alpha / S) * ((S + 1).digamma() - (alpha + 1).digamma())).sum(dim=1) # aleatoric uncertainty
        Udist = entropy - Udata # epistemic uncertainty
        if reduce:
            Udata = Udata.sum() / alpha.shape[0]
            Udist = Udist.sum() / alpha.shape[0]
        return Udata, Udist    
    elif UNCERTAINTY_mode == 'evidence':
        Vac, Diss = getDisn(alpha)
        return Diss, Vac
    else:
        raise NotImplementedError(f'Uncertainty not implemented: {UNCERTAINTY_mode}')