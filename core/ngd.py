import torch
from torch.optim.optimizer import Optimizer, required
import copy as cp
import torch.optim as optim

class NGD(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)

        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(NGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, whitening_matrices = None, closure=None):
        """Performs a single optimization step.
        Arguments:
            whitening_matrices(optional): dictionary of whitening matrices
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            whitening_matrices_list = []
            if whitening_matrices != None:
                whitening_matrices_list_values = list(whitening_matrices.values())
                whitening_matrices_list_keys = list(whitening_matrices.keys())

            idx = 0
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if whitening_matrices_list_values and not len(d_p.shape) == 1:
                    #print('idx = {}, whitening_matrices[{}].shape, whitening_matrices[{}].shape'.format(idx, whitening_matrices_list_keys[2*idx],whitening_matrices_list_keys[2*idx-1]))
                    psi = whitening_matrices_list_values[2 * idx + 1]
                    gamma = whitening_matrices_list_values[2 * idx]
                    #print('type d_p = {}, psi = {}, gamma = {}'.format(type(d_p), type(psi), type(gamma)))
                    #print('Shape d_p = {}, psi = {}, gamma = {}'.format(d_p.shape, psi.shape, gamma.shape))
                    #print('NGD: psi_tensor max = {}, min = {}'.format(torch.max(psi.flatten()), torch.min(psi.flatten())))
                    #print('NGD: gamma_tensor max = {}, min = {}'.format(torch.max(gamma.flatten()), torch.min(gamma.flatten())))

                    psi_tensor = psi#torch.from_numpy(psi.astype(np.float32))
                    gamma_tensor = gamma#torch.from_numpy(gamma.astype(np.float32))
                    sz = cp.deepcopy(d_p.shape)
                    tensor_dims = len(d_p.shape)
                    if tensor_dims == 4:
                        d_p = torch.reshape(d_p, (sz[0], sz[1]* sz[2]* sz[3]))

                    d_p = psi_tensor @ d_p @gamma_tensor
                    if tensor_dims == 4:
                        d_p = torch.reshape(d_p, sz)
                    idx = idx + 1
                '''
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf
                '''
                p.add_(d_p, alpha=-group['lr'])

        return loss

def select_optimizer(model, optimizer_arg, lr):
    if optimizer_arg == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_arg == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
    elif optimizer_arg == 'ngd':
        optimizer = NGD(model.parameters(), lr=lr)
    return optimizer

