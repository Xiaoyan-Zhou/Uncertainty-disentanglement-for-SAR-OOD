import torch
import torch.nn.functional as F

def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -30, 30))#  torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None, loglikelihood_var_flag=False):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    # dawn 论文Revisiting Essential and Nonessential Settings of Evidential Deep Learning中说明var那项也是非必要的，
    # loglikelihood = loglikelihood_err + loglikelihood_var 
    if loglikelihood_var_flag:
        return loglikelihood_err + loglikelihood_var
    else:
        return loglikelihood_err  

### dawn when KL=False and loglikelihood_var_flag=False, the loss function is the same as Revisiting Essential and Nonessential Settings of Evidential Deep Learning
def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, KL=False, loglikelihood_var_flag=False):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device, loglikelihood_var_flag=loglikelihood_var_flag)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    if KL:
        return loglikelihood + kl_div
    else:
        # dawn : Revisiting Essential and Nonessential Settings of Evidential Deep Learning 论文中证明KL散度的存在是非必要的
        return loglikelihood 

def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None, KL_flag=False, loglikelihood_var_flag=False):
    if not device:
        device = get_device()
    evidence = relu_evidence(output) # dawn use relu function get evidence, original function in paper EDL
    # evidence = exp_evidence(output) # dawn revised in our method

    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device, KL = KL_flag, loglikelihood_var_flag=loglikelihood_var_flag)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def MEDL(labels_1hot, evidence):
    lamb1 = 1.0
    lamb2 = 1.0
    num_classes = evidence.shape[-1]

    gap = labels_1hot - (evidence + lamb2) / \
            (evidence + lamb1 * (torch.sum(evidence, dim=-1, keepdim=True) - evidence) + lamb2 * num_classes)

    loss_mse = gap.pow(2).sum(-1)

    return loss_mse.mean()