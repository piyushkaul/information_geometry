import torch.nn as nn
from collections import OrderedDict
import torch
import numpy as np
import inspect
import matplotlib.pyplot as plt
import math
import itertools
from sklearn.random_projection import johnson_lindenstrauss_min_dim

class Hook():
    enable = True
    def __init__(self, module, name=None, backward=False):
        self.backward = backward
        if isinstance(module,nn.Conv2d):
            self.fwd_index_in = 0
            self.fwd_index_out = 0
            self.back_index_in = 0
            self.back_index_out = 0
            self.stride = module.stride
            self.kernel_size = module.kernel_size
            self.padding = module.padding
            self.module_type = 'Conv2d'
        elif isinstance(module,nn.Linear):
            self.fwd_index_in = 0
            self.fwd_index_out = 0
            self.back_index_in = 0
            self.back_index_out = 0
            self.module_type = 'Linear'
            self.kernel_size = 1
            self.padding = 0
            self.stride = 1
        else:
            raise Exception('Unsupported Layer Type')

        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)
        self.name = name


    def print_tuple_or_tensor(self, tt, tag):
        #if tt is None:
        #    print('input is none')
        #    return
        print('tag = {}, backward = {}'.format(tag, self.backward))

        if self.backward:
            direction = 'backward'
        else:
            direction = 'forward'

        if isinstance(tt, tuple):
            print('direction = {}, name: {}, shape : [ m {}'.format(direction, tag, len(tt)))
            for tn, itt in enumerate(tt):
                #self.print_tuple_or_tensor(itt, tag + str(tn))
                if itt == None:
                    continue
                    raise Exception('itt is none')
                print('direction = {}, name: {}, shape : {}'.format(direction, tag + str(tn), itt.shape))
            print(']')
        else:
            print('direction = {}, name: {}, shape : {}'.format(direction, tag, tt.shape))

    def get_first_element(self, item, index):
        if isinstance(item, tuple):
            #print('get first element = {}'.format(item[index].shape))
            return item[index].detach()
        else:
            #print('get first element = {}'.format(item[index].shape))
            return item.detach()


    def hook_fn(self, module, input, output):
        #self.print_tuple_or_tensor(input, self.name + '_input')
        #self.print_tuple_or_tensor(output,  self.name + '_output')

        if self.backward == False:
            param = self.get_first_element( input, self.fwd_index_in)
            #self.output = self.get_first_element( output, self.fwd_index_out)
        else:
            #self.input = self.get_first_element( input, self.back_index_in)
            param = self.get_first_element( output, self.back_index_out)

        if self.module_type == 'Conv2d' and self.backward == False:
            kernel_size = self.kernel_size
            padding = self.padding
            stride = self.stride
            #reduced_param = im2col_indices(param, kernel_size[0], kernel_size[1], padding[0], stride[0])
            reduced_param = torch.nn.functional.unfold(param, (kernel_size[0], kernel_size[1]), stride=1)
            reduced_param = reduced_param.transpose(1, 2)
            reduced_param = reduced_param.reshape(-1, reduced_param.shape[-1])
            #reduced_param = reduced_param.T
            self.spatial_sizes = 1.0/(param.shape[2] * param.shape[3])
            self.batch_sizes = param.shape[0]
            #reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param, axis=0, keepdims=True)
            num_samples = min(reduced_param.shape[0], 200)
            random_samples = torch.randint(0, reduced_param.shape[0], (num_samples,))
            reduced_param =  reduced_param[random_samples, :]
        elif self.module_type == 'Conv2d' and self.backward == True:
            sz = param.shape
            reduced_param = param.permute(1,0,2,3)
            reduced_param = torch.reshape(reduced_param, (sz[1],-1))
            reduced_param = reduced_param.T
            self.spatial_sizes = (param.shape[2] * param.shape[3])
            self.batch_sizes = reduced_param.shape[0]
            #reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param,axis=0, keepdims=True)
            num_samples = min(reduced_param.shape[0], 200)
            random_samples = torch.randint(0, reduced_param.shape[0], (num_samples,))
            reduced_param =  reduced_param[random_samples, :]

        elif self.module_type == 'Linear':
            reduced_param = param
            self.spatial_sizes = 1
            self.batch_sizes = param.shape[0]
        else:
            raise Exception('Unknown Module type = {}'.format(self.module_type))


        if self.backward == False:
            self.input = reduced_param
            # self.output = self.get_first_element( output, self.fwd_index_out)
        else:
            # self.input = self.get_first_element( input, self.back_index_in)
            self.output = reduced_param

    def close(self):
        self.hook.remove()



class ModelFIM(nn.Module):
    def __init__(self, args):
        super(ModelFIM, self).__init__()
        self.subspace_fraction = args.subspace_fraction
        #self.GS = OrderedDict() #dummy here
        #self.common_init()

    def get_whitening_matrices(self):
        matrix_list = []
        for key, val in self.GSINV.items():
            matrix_list.append(val)
        return matrix_list

    def register_hooks_(self, model, hookF, hookB):
        #print('register hooks called')
        #print('----------------------START--------------------')
        #print(model)
        #print('-----------------------END---------------------')
        for name, layer in list(model._modules.items()):
            #print('register_hooks: layer name = {}'.format(name))
            tl = type(layer)
            print(tl)
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                hookF.append(Hook(layer,  name=name))
                hookB.append(Hook(layer,  name=name, backward=True))
            if not(len(layer._modules.items()) == 0):
                #print('register_hooks: layer name recursion')
                hookF, hookB = self.register_hooks_(layer, hookF, hookB)
        return hookF, hookB

    # hookB = [Hook(layer[1],backward=True) for layer in list(model._modules.items())]
    def register_hooks(self):
        self.hookF, self.hookB = self.register_hooks_(self, [], [])

    def print_hooks(self):
        hookForward = self.hookF
        hookBack = self.hookB
        print('***' * 3 + '  Forward Hooks Inputs & Outputs  ' + '***' * 3)
        for hook in hookForward:
            print(hook.name)
            print(hook.input[0].shape)
            print(hook.output.shape)
            print('---' * 17)
        print('\n')
        print('***' * 3 + '  Backward Hooks Inputs & Outputs  ' + '***' * 3)
        for hook in hookBack:
            print(hook.name)
            print(hook.input.shape)
            print(hook.output[0].shape)
            print('---' * 17)

    def get_act_and_grads(self):
        hookForward = self.hookF
        hookBack = self.hookB
        params = []
        for (hf, hb) in zip(hookForward, hookBack):
            params.append(hf.input)
            params.append(hb.output)
        return params

    def initialize_matrices(self, params):
        print('num_params = {}'.format(len(params)))
        for item_no, item in enumerate(params):
            if item_no%2 == 0:
                name = 'PSI' + str(item_no//2) + '_AVG'
            else:
                name = 'GAM' + str(item_no//2) + '_AVG'
            depth = item.shape[1]
            self.GS[name] = torch.eye((depth), device=self.device)
        for item_no, (key, val) in enumerate(self.GS.items()):
            self.GSLOWER[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0], item_no==0), device=self.device)
            self.GSLOWERINV[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0], item_no==0), device=self.device)
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = self.GS[key]
            subspace_size = self.get_subspace_size(self.GSINV[key].shape[0], item_no==0)
            eigvec_subspace = self.GS[key][:, -subspace_size:]
            self.P[key] = eigvec_subspace
            self.corr_curr[key] = None
            self.corr_curr_lower_proj[key] = None
            self.corr_curr_lower[key] = None
            self.spatial_sizes[key] = 1
            self.batch_sizes[key] = 1
            self.damping[key] = 1
            self.params[key] = None
            if item_no%2==0:
                self.eigval_f_inv_list_per_epoch.append([])
        self.track_gs('initialize_matrices')

    def common_init(self, args, hook_enable=True, logger=None):
        self.use_cuda = not args.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.GS = OrderedDict()
        self.GSLOWER = OrderedDict()
        self.GSLOWERINV = OrderedDict()


        self.GSINV = OrderedDict()

        self.P = OrderedDict()
        self.corr_curr = OrderedDict()
        self.corr_curr_lower_proj = OrderedDict()
        self.corr_curr_lower = OrderedDict()
        self.spatial_sizes = OrderedDict()
        self.batch_sizes = OrderedDict()
        self.eigval_f_inv_list_per_epoch = [] #list[epoch] of list[per layer] of eigval arrays for all epochs.
        self.eigval_f_inv_list = [] #list [per layer] of eigval arrays current epoch
        self.damping = OrderedDict()
        self.params = OrderedDict()
        self.tick = 0
        self.gamma = 0.1
        self.epoch_no = 0
        self.first_time = True
        self.logger = logger


        if args.random_projection:
            self.random_projection = True
            self.get_subspace_size = self.get_subspace_size_fraction#get_subspace_size_random
        else:
            self.random_projection = False
            self.get_subspace_size = self.get_subspace_size_fraction
        self.dump_eigenvalues = args.dump_eigenvalues
        self.matrices_initialized = False
        if hook_enable:
            self.register_hooks()

    #def epoch_trace_det_dump(self):
    #    for item_no, item in enumerate(self.eigval_f_inv_list_per_epoch):

    def epoch_bookkeeping(self):
        for item_no, item in enumerate(self.eigval_f_inv_list_per_epoch):
            features = np.sort(self.eigval_f_inv_list[item_no]).tolist()
            features.insert(0,self.epoch_no)
            if self.epoch_no == 0:
                header = ['epoch']
                for itr in range(len(features)-1):
                    header.append('COL'+str(itr))
                self.eigval_f_inv_list_per_epoch[item_no].append(header)
            self.eigval_f_inv_list_per_epoch[item_no].append(features)
        self.epoch_no = self.epoch_no + 1

    def dump_eigval_arrays(self):
        import csv
        for list_idx, list_item in enumerate(self.eigval_f_inv_list_per_epoch):
            with open('eigval_arrays' + str(list_idx) + '.csv', 'w', newline='') as f:
                csvwriter = csv.writer(f)
                csvwriter.writerows(list_item)

    def update_eigval_arrays(self):
        if self.dump_eigenvalues:
            gs_values = list(self.GS.values())
            self.eigval_f_inv_list = []
            for item_no, (key, item) in enumerate(self.GS.items()):
                 if item_no%2==0:
                     eigval_psi, eigvec_psi = torch.symeig(gs_values[item_no], eigenvectors=False)
                     eigval_gam, eigvec_gam = torch.symeig(gs_values[item_no+1], eigenvectors=False)
                     if self.use_cuda:
                         eigval_f_inv = np.kron(eigval_psi.cpu().numpy(), eigval_gam.cpu().numpy())
                     else:
                         eigval_f_inv = np.kron(eigval_psi.numpy(), eigval_gam.numpy())
                     #print('eigval_f_inv = {}'.format(eigval_f_inv))
                     self.eigval_f_inv_list.append(eigval_f_inv)
                     self.logger.log_eigvals(eigval_f_inv, item_no//2, self.tick)
                 #    plt.plot(range(len(eigval_f_inv)), eigval_f_inv, 'r', label='test loss')
                 #    plt.xlabel('Eigenvalues')
                 #    plt.ylabel('Loss')
                 #    plt.legend(loc=2, fontsize="small")
                 #    plt.show()

    def check_nan(self, tensor, message=None):
        if torch.isnan(tensor).any():
            raise Exception('Got a Nan for tensor = {}', message)

    def track_gs(self, func='dummy', dict_to_use=None):
        if not dict_to_use:
            dict_to_use = self.GS
        for item_no, (key, item) in enumerate(dict_to_use.items()):
            print('{} : GS[{}] SHAPE={}'.format(func, key, dict_to_use[key].shape))

        #for item_no, (key, item) in enumerate(dict_to_use.items()):
        #    print('{} : GS[{}] MAX = {}, MIN = {}, multiplier = {}'.format(func, key, dict_to_use[key].flatten().max(), dict_to_use[key].flatten().min(), self.spatial_sizes[key] * self.batch_sizes[key]))

    def forward(self, X):
        raise Exception('Inherit this class')


    def get_subspace_size_random(self, full_space_size, first_item=False, tag=None):
        subspace_size = johnson_lindenstrauss_min_dim(n_samples=full_space_size, eps=0.9)
        if subspace_size > full_space_size:
            subspace_size = full_space_size

        if first_item:
            subspace_size = full_space_size
 
         

        if self.first_time:
            print('For Layer = {}, Full Space size = {}, Subspace Size = {}'.format(tag, full_space_size, subspace_size))
     


        return int(subspace_size)

    def get_subspace_size_fraction(self, full_space_size, first_item=False, tag=None):
        subspace_size = int(full_space_size * self.subspace_fraction)
        if subspace_size < 64:
            subspace_size = full_space_size

        if first_item:
            subspace_size = full_space_size

        if full_space_size == 784:
            subspace_size = full_space_size

        if self.first_time:
            print('For Layer = {}, Full Space size = {}, Subspace Size = {}'.format(tag, full_space_size, subspace_size))
        return subspace_size

    def get_grads(self):
        #self.print_hooks()
        params = self.get_act_and_grads()
        if not self.matrices_initialized:
            self.initialize_matrices(params)
            self.matrices_initialized = True
        return params


    def projection_matrix_update(self, params):
        if self.random_projection:
            self.random_projection_matrix_update(params)
        else:
            self.orthogonal_projection_matrix_update(params)
        self.first_time = False


    def random_projection_matrix_update(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            #eigval, eigvec = torch.symeig(self.GS[key], eigenvectors=True)
            num_components = params[item_no].shape[0]
            num_features = params[item_no].shape[1]
            subspace_size = self.get_subspace_size(num_features, item_no==0, tag=key)
            P = np.random.normal(loc=0, scale=1.0 / np.sqrt(subspace_size), size=(subspace_size, num_features))
            self.P[key] = torch.from_numpy(P.T.astype(np.float32)).to(self.device)
            #print('random_projection_matrix_update: size of P[{}] = {}'.format(key, self.P[key].shape))


    def orthogonal_projection_matrix_update(self, params):
        if self.subspace_fraction == 1:
            return
        #gs_keys=list(self.GS.keys())


        for item_no, (key, item) in enumerate(self.GS.items()):
            eigval, eigvec = torch.symeig(self.GS[key], eigenvectors=True)
            subspace_size = self.get_subspace_size(eigvec.shape[0], item_no==0, tag=key)
            eigvec_subspace = eigvec[:, -subspace_size:]
            self.P[key] = eigvec_subspace
            #print('orthogonal_projection_matrix_update: size of P[{}] = {}'.format(key, self.P[key].shape))

    def project_vec_to_lower_space(self, matrix, key):
        #print('project_to_lower_space: Shape of P[{}] = {}. Shape of matrix = {}'.format(key, self.P[key].shape, matrix.shape))
        if self.subspace_fraction==1:
            return matrix
        return matrix @ self.P[key]

    def project_vec_to_higher_space(self, matrix, key):
        #print('project_to_higher_space: Shape of P[{}] = {}. Shape of matrix = {}'.format(key, self.P[key].shape, matrix.shape))
        if self.subspace_fraction == 1:
            return matrix
        return self.P[key] @ matrix

    def project_mtx_To_higher_space(self, matrix, key):
        #print('self.P[{}].shape = {}, matrix.shape = {}'.format(key, self.P[key].shape, matrix.shape))
        if self.subspace_fraction == 1:
            return matrix
        return self.P[key] @ matrix @ self.P[key].T

    # def matrix_inv_lemma(self, X, GS, key='none'):
    #     num_batches = X.shape[0]
    #     # cinv = ainv - ainv * x * inv(eye(32) + x' * ainv * x ) * x' * ainv
    #     #print('X.shape = {}, GSINV[{}].shape = {}
    #     alpha = 0.99
    #     GS = GS * (1.0/alpha)
    #     X = X * (math.sqrt(1-alpha))
    #     inner_term = torch.eye(num_batches, device=self.device)*(1/(1-alpha)) + X @ GS @ X.T
    #     xg = X @ GS
    #     gx = GS @ X.T
    #     GS = GS - gx @ torch.inverse(inner_term) @ xg
    #     #print(
    #     #    'X.shape = {}, GSINV[{}].shape = {}, inner_term.shape = {}'.format(X.shape, key, self.GSLOWERINV[key].shape,
    #     #                                                                       inner_term.shape))
    #     return GS
    # def matrix_inv_lemma(self, X, GS, device=None, alpha=0.9, key=None):
    #     alpha= self.gamma * self.damping[key]
    #     alpha = 1
    #     num_batches = X.shape[0]
    #     GS = GS# * (1 / alpha) #+ torch.eye(GS.shape[0], device=device) * 0.1
    #     id_mat = torch.eye(num_batches, device=device)# * (1 / (1 - alpha))
    #     inner_term = id_mat + X @ GS @ X.T
    #     xg = X @ GS
    #     gx = GS @ X.T
    #     GS = GS - gx @ torch.inverse(inner_term) @ xg
    #     return GS

    def matrix_inv_lemma(self, X, GS, device=None, alpha=0.99, key=None, lr=1):
        beta = 1 - alpha
        beta2 = math.sqrt(beta)
        num_batches = X.shape[1]
        id_mat = torch.eye(num_batches, device=device)

        inner_term = id_mat + (beta2 * X.T) @ (GS / alpha) @ (X * beta2)
        gx = (1 / alpha * GS) @ (X * beta2)
        xg = (beta2 * X.T) @ (GS * 1 / alpha)
        GS = (1 / alpha) * GS - gx @ torch.inverse(inner_term) @ xg #+ (1+np.sqrt(lr[0]))* torch.eye(GS.shape[0], device=device)
        return GS

    def test_matrix_inv_lemma(self):
        param = torch.normal(0, 1, size=(250, 64), device=self.device)
        mat = torch.normal(0,1,size=(250,250), device=self.device)
        mat2 = 0.99 * mat + 0.01 * param @ param.T;
        mat_inv = torch.inverse(mat)
        mat_inv2 = torch.inverse(mat2)
        mat_inv2_mil = self.matrix_inv_lemma(param, mat_inv)
        print('max difference = {}'.format(torch.max(mat_inv2 - mat_inv2_mil)))
        


    def maintain_params(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            if item_no%2 == 0:
                self.params[key] = params[item_no].clone() / np.sqrt(params[item_no].shape[0])
            else:
                self.params[key] = params[item_no].clone()


    def maintain_params_lower(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            if item_no%2 == 0:
                self.params[key] = params[item_no].clone() / np.sqrt(params[item_no].shape[0])
            else:
                self.params[key] = params[item_no].clone()


    def maintain_corr(self, params):
        #self.track_gs('maintain_corr before')

        for item_no, (key, item) in enumerate(self.GS.items()):

            if item_no%2 == 0:
                self.corr_curr[key] = params[item_no].T  @ (params[item_no] / params[item_no].shape[0])
            else:
                self.corr_curr[key] = params[item_no].T  @ params[item_no]
            #self.check_nan(self.corr_curr[item_no], message=key)
            #multiplier = self.spatial_sizes[key]
            #params_temp = params[item_no] * multiplier
            #self.corr_curr[key] = params_temp.T @ (params_temp * 1 / (self.batch_sizes[key]))
            #if key=='GAM0_AVG':
            #    with open('GAM0_AVG','w') as fid:
            #        params[item_no].tofile(fid,'\n','%f')
            #print('MAX(param[{}]) = {}, MAX(corr_curr[{}]) = {}, spatial_size = {}'.format(key, params[item_no].flatten().max(), key, self.corr_curr[key].flatten().max(), self.spatial_sizes[key]))
            #print('param[{}].shape = {}, corr_curr[{}] = {}'.format(item_no, params[item_no].shape, key, self.corr_curr[key].shape))
            #if np.isnan(self.corr_curr[key]).any():
            #    print(params[item_no])
            #    raise Exception('key={} is nan'.format(key))

        #self.track_gs('maintain_corr after', dict_to_use=self.corr_curr)


    def maintain_corr_lower(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.corr_curr_lower_proj[key] = self.project_vec_to_lower_space(params[item_no], key)
            if item_no%2 == 0:
                self.corr_curr_lower[key] = self.corr_curr_lower_proj[key].T @ (self.corr_curr_lower_proj[key] / self.corr_curr_lower_proj[key].shape[0])
            else:
                self.corr_curr_lower[key] = self.corr_curr_lower_proj[key].T @ self.corr_curr_lower_proj[key]

        #self.track_gs('maintain_corr after', dict_to_use=self.corr_curr_lower)

    def maintain_avgs(self):
        #self.track_gs('maintain_avgs before')
        alpha = min(1 - 1 / (self.tick+1), 0.95)
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GS[key] = alpha * self.GS[key] + (1 - alpha) * self.corr_curr[key]
            #self.check_nan(self.GS[key], message=key)
        #self.track_gs('maintain_avgs after')

    def maintain_avgs_lower(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, self.corr_curr_lower[key].shape, key, self.GSLOWER[key].shape))
            self.GSLOWER[key] = alpha * self.GSLOWER[key] + (1 - alpha) * self.corr_curr_lower[key]


    def reset_invs(self):
        for item_no, (key, item) in enumerate(self.GSINV.items()):
            self.GSINV[key] = torch.eye(self.GSINV[key].shape[0], device=self.device)

    def reset_all(self):
        for item_no, (key, item) in enumerate(self.GSINV.items()):
            self.GSINV[key] = torch.eye(self.GSINV[key].shape[0], device=self.device)
            self.GSLOWER[key]  = torch.eye(self.GSLOWER[key].shape[0], device=self.device)
            self.GS[key] = torch.eye(self.GS[key].shape[0], device=self.device)
            self.GSLOWERINV[key] = torch.eye(self.GSLOWERINV[key].shape[0], device=self.device)
            subspace_size = self.get_subspace_size(self.GSINV[key].shape[0], item_no==0)
            eigvec_subspace = self.GS[key][:, -subspace_size:]
            self.P[key] = eigvec_subspace


    def get_invs_recursively(self, lr):
        self.get_damping_factor(self.GS)
        for item_no, (key, item) in enumerate(self.GS.items()):
            #XFULL = self.corr_curr[key]
            XFULL = self.params[key]
            #print('XFULL SHAPE = {}'.format(XFULL.shape))
            self.GSINV[key] = self.matrix_inv_lemma(XFULL.T, self.GSINV[key], device=self.device, key=key, lr=lr)

    def get_invs_recursively_lower(self, lr):
        for item_no, (key, item) in enumerate(self.GS.items()):
            XLOWER = self.corr_curr_lower_proj[key]
            #print('XLOWER SHAPE = {}'.format(XLOWER.shape))
            self.GSLOWERINV[key] = self.matrix_inv_lemma(XLOWER.T, self.GSLOWERINV[key], device=self.device, key=key, lr=lr)
            self.GSINV[key] = self.project_mtx_To_higher_space(self.GSLOWERINV[key], key)


    def get_inverses_direct(self):
        self.get_damping_factor(self.GS)
        for item_no, (key, item) in enumerate(self.GS.items()):
            #eigval, eigvec = torch.symeig(self.GS[key], eigenvectors=True)
            #med_eig = torch.median(eigval)
            #eigval[eigval<med_eig] = med_eig
            #self.GS[key] = eigvec @ torch.diag(eigval) @ eigvec.T
            self.GSINV[key] = torch.inverse(self.GS[key] + torch.eye(self.GS[key].shape[0], device=self.device) * self.gamma * self.damping[key])
            #self.check_nan(self.GSINV[key], message=key)

        #self.track_gs('get_inverses_direct after', dict_to_use=self.GSINV)

    def get_damping_factor(self, matrices):
        items = list(matrices.items())
        for item_no, item in enumerate(items):
            if item_no%2 == 0:
                self.damping[item[0]] = torch.sqrt(torch.trace(items[item_no][1])/torch.trace(items[item_no+1][1]))
            else:
                self.damping[item[0]] = torch.sqrt(torch.trace(items[item_no][1])/torch.trace(items[item_no-1][1]))

    def get_inverses_direct_lower(self):
        self.get_damping_factor(self.GSLOWER)
        for item_no, (key, item) in enumerate(self.GS.items()):
            GSPROJINV = torch.inverse(self.GSLOWER[key] + torch.eye(self.GSLOWER[key].shape[0], device=self.device) * self.gamma * self.damping[key])
            self.GSINV[key] = self.project_mtx_To_higher_space(GSPROJINV, key)

    def maintain_invs(self, params, args, lr=1):
        tick = self.tick

        if args.inv_type == 'recursive' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            self.maintain_params(params)
            self.get_invs_recursively(lr)
            if tick % args.inv_period == 0:
                self.get_inverses_direct()

        elif args.inv_type == 'recursive' and args.subspace_fraction < 1:
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            self.maintain_params(params)
            self.get_invs_recursively_lower(lr)
            if tick % args.inv_period == 0:
                self.get_inverses_direct_lower()

        elif args.inv_type == 'direct' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            if tick % args.inv_period == 0:
                self.get_inverses_direct()
        elif args.inv_type == 'direct' and args.subspace_fraction < 1:
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            if tick % args.inv_period == 0:
                self.get_inverses_direct_lower()
        else:
            raise Exception('unknown combination')

        if tick % args.inv_period == 0:
            self.update_eigval_arrays()

        self.tick = self.tick + 1

    def actual_loss_classification(self, output, criterion):
        max_class = torch.argmax(output, 1)
        loss = criterion(output, max_class)
        loss.backward(retain_graph=True)
    
   
    def actual_loss_mse(self, output, criterion):
        #max_class = torch.argmax(output, 1)
        loss = criterion(output, output)
        loss.backward(retain_graph=True)

    def maintain_fim(self, args, batch_idx, type_of_loss=False, output=None, criterion=None, lr=1):
        if type_of_loss:
            if type_of_loss == 'classification':
                self.actual_loss_classification(output,criterion)
            elif type_of_loss == 'autoencoder':
                self.actual_loss_mse(output, criterion)
          
        params = self.get_grads()
        self.maintain_invs(params, args, lr=lr)
        if batch_idx % args.proj_period == 0:
            self.projection_matrix_update(params)
