r"""Functional interface"""
import math
import torch
from torch import Tensor
from typing import List, Optional
import copy as cp


def whiten(whitening_matrices_list_values, idx, d_p):
    psi_tensor = whitening_matrices_list_values[2 * idx + 1]
    gamma_tensor = whitening_matrices_list_values[2 * idx]

    sz = cp.deepcopy(d_p.shape)
    tensor_dims = len(d_p.shape)
    if tensor_dims == 4:
        d_p = torch.reshape(d_p, (sz[0], sz[1] * sz[2] * sz[3]))

    d_p = psi_tensor @ d_p @ gamma_tensor
    if tensor_dims == 4:
        d_p = torch.reshape(d_p, sz)

    return d_p


def adam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         beta1: float,
         beta2: float,
         lr: float,
         weight_decay: float,
         eps: float,
         whitening_matrices: List[Tensor]):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    """
    idx = 0
    for i, param in enumerate(params):

        grad = grads[i]

        if whitening_matrices and not len(grad.shape) == 1:
            grad = whiten(whitening_matrices, idx, grad)
            idx = idx + 1

        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # Decay the first and second moment running average coefficient
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        step_size = lr / bias_correction1

        param.addcdiv_(exp_avg, denom, value=-step_size)

def sgd(params: List[Tensor],
        d_p_list: List[Tensor],
        momentum_buffer_list: List[Optional[Tensor]],
        weight_decay: float,
        momentum: float,
        lr: float,
        dampening: float,
        nesterov: bool,
        whitening_matrices: List[Tensor]):
    r"""Functional API that performs SGD algorithm computation.
    See :class:`~torch.optim.SGD` for details.
    """

    for i, param in enumerate(params):

        d_p = d_p_list[i]

        if whitening_matrices and not len(d_p.shape) == 1:
            d_p = whiten(whitening_matrices, i, d_p)

        if weight_decay != 0:
            d_p = d_p.add(param, alpha=weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(d_p).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

            if nesterov:
                d_p = d_p.add(buf, alpha=momentum)
            else:
                d_p = buf

        param.add_(d_p, alpha=-lr)

