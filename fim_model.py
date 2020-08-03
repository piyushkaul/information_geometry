import torch.nn as nn
from collections import OrderedDict
import torch
import numpy as np
import inspect
import matplotlib.pyplot as plt


class ModelFIM(nn.Module):
    def __init__(self, subspace_fraction=0.1):
        super(ModelFIM, self).__init__()
        self.subspace_fraction = subspace_fraction
        #self.GS = OrderedDict() #dummy here
        #self.common_init()

    def common_init(self):
        self.GSLOWER = OrderedDict()
        self.GSLOWERINV = OrderedDict()
        for key, val in self.GS.items():
            self.GSLOWER[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0]))
            self.GSLOWERINV[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0]))

        self.GSINV = OrderedDict()

        self.P = OrderedDict()
        self.corr_curr = OrderedDict()
        self.corr_curr_lower_proj = OrderedDict()
        self.corr_curr_lower = OrderedDict()
        self.spatial_sizes = OrderedDict()
        self.batch_sizes = OrderedDict()
        self.eigval_f_inv_list_per_epoch = [] #list[epoch] of list[per layer] of eigval arrays for all epochs.
        self.eigval_f_inv_list = [] #list [per layer] of eigval arrays current epoch

        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = self.GS[key]
            subspace_size = self.get_subspace_size(self.GSINV[key].shape[0])
            eigvec_subspace = self.GS[key][:, -subspace_size:]
            self.P[key] = eigvec_subspace
            self.corr_curr[key] = None
            self.corr_curr_lower_proj[key] = None
            self.corr_curr_lower[key] = None
            self.spatial_sizes[key] = 1
            self.batch_sizes[key] = 1
            if item_no%2==0:
                self.eigval_f_inv_list_per_epoch.append([])

        self.tick = 0
        self.gamma = 0.1
        self.epoch_no = 0


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

    def check_nan(self, tensor, message=None):
        if torch.isnan(tensor).any():
            raise Exception('Got a Nan for tensor = {}', message)
        #print('For Tensor {}: max={}'.format(message, torch.max(tensor.flatten())))
        #max_eig = LA.cond(tensor.numpy(), 2)
        #min_eig = LA.cond(tensor.numpy(), -2)
        #cond_num = max_eig/min_eig
        #print('COND NUM: Caller={}, For Tensor {}: condition number={}, max_eig={}, min_eig={}'.format(inspect.stack()[1].function, message, cond_num, max_eig, min_eig ))

        #cond_num = LA.cond(self.GS[key].numpy(), 2) / LA.cond(self.GS[key].numpy(), -2)
        #if cond_num > 10:
        #    self.GS[key] = torch.eye(self.GS[key].shape[0])

    def track_gs(self, func, dict_to_use=None):
        return
        if not dict_to_use:
            dict_to_use = self.GS
        for item_no, (key, item) in enumerate(dict_to_use.items()):
            print('{} : GS[{}] MAX = {}, MIN = {}, multiplier = {}'.format(func, key, dict_to_use[key].flatten().max(), dict_to_use[key].flatten().min(), self.spatial_sizes[key] * self.batch_sizes[key]))

    def forward(self, X):
        raise Exception('Inherit this class')

    def get_subspace_size(self, full_space_size):
        subspace_size = int(full_space_size * self.subspace_fraction)
        if subspace_size < 64:
            subspace_size = full_space_size
        return subspace_size

    def get_grads(self):
        raise Exception('Inherit this class')

    def projection_matrix_update(self):
        if self.subspace_fraction == 1:
            return
        #gs_keys=list(self.GS.keys())
        gs_values = list(self.GS.values())
        self.eigval_f_inv_list = []
        for item_no, (key, item) in enumerate(self.GS.items()):
             if item_no%2==0:
                 eigval_psi, eigvec_psi = torch.symeig(gs_values[item_no], eigenvectors=False)
                 eigval_gam, eigvec_gam = torch.symeig(gs_values[item_no+1], eigenvectors=False)
                 eigval_f_inv = np.kron(eigval_psi.numpy(), eigval_gam.numpy())
                 #print('eigval_f_inv = {}'.format(eigval_f_inv))
                 self.eigval_f_inv_list.append(eigval_f_inv)
             #    plt.plot(range(len(eigval_f_inv)), eigval_f_inv, 'r', label='test loss')
             #    plt.xlabel('Eigenvalues')
             #    plt.ylabel('Loss')
             #    plt.legend(loc=2, fontsize="small")
             #    plt.show()

        for item_no, (key, item) in enumerate(self.GS.items()):
            eigval, eigvec = torch.symeig(self.GS[key], eigenvectors=True)
            subspace_size = self.get_subspace_size(eigvec.shape[0])
            eigvec_subspace = eigvec[:, -subspace_size:]
            self.P[key] = eigvec_subspace

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

    def matrix_inv_lemma(self, X, GS, key='none'):
        num_batches = X.shape[0]
        # cinv = ainv - ainv * x * inv(eye(32) + x' * ainv * x ) * x' * ainv
        #print('X.shape = {}, GSINV[{}].shape = {}'.format(X.shape, key, GS.shape))
        inner_term = torch.eye(num_batches) + X @ GS @ X.T
        xg = X @ GS
        gx = GS @ X.T
        GS = GS - gx @ torch.inverse(inner_term) @ xg
        #print(
        #    'X.shape = {}, GSINV[{}].shape = {}, inner_term.shape = {}'.format(X.shape, key, self.GSLOWERINV[key].shape,
        #                                                                       inner_term.shape))
        return GS

    def maintain_corr(self, params):
        self.track_gs('maintain_corr before')

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

        self.track_gs('maintain_corr after', dict_to_use=self.corr_curr)


    def maintain_corr_lower(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.corr_curr_lower_proj[key] = self.project_vec_to_lower_space(params[item_no], key)
            self.corr_curr_lower[key] = self.corr_curr_lower_proj[key].T @ self.corr_curr_lower_proj[key]
        self.track_gs('maintain_corr after', dict_to_use=self.corr_curr_lower)

    def maintain_avgs(self):
        self.track_gs('maintain_avgs before')
        alpha = min(1 - 1 / (self.tick+1), 0.95)
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GS[key] = alpha * self.GS[key] + (1 - alpha) * self.corr_curr[key]
            self.check_nan(self.GS[key], message=key)
        self.track_gs('maintain_avgs after')

    def maintain_avgs_lower(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GSLOWER[key] = alpha * self.GSLOWER[key] + (1 - alpha) * self.corr_curr_lower[key]


    def get_invs_recursively(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            XFULL = self.corr_curr[item_no]
            self.GSINV[key] = self.matrix_inv_lemma(XFULL, self.GSINV[key], key=key)

    def get_invs_recursively_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            XLOWER = self.corr_curr_lower_proj[key]
            self.GSLOWERINV[key] = self.matrix_inv_lemma(XLOWER, self.GSLOWERINV[key], key=key)


    def get_inverses_direct(self):
        self.get_damping_factor()
        for item_no, (key, item) in enumerate(self.GS.items()):
            #eigval, eigvec = torch.symeig(self.GS[key], eigenvectors=True)
            #med_eig = torch.median(eigval)
            #eigval[eigval<med_eig] = med_eig
            #self.GS[key] = eigvec @ torch.diag(eigval) @ eigvec.T
            self.GSINV[key] = torch.inverse(self.GS[key] + torch.eye(self.GS[key].shape[0]) * self.gamma * self.damping[key])
            self.check_nan(self.GSINV[key], message=key)

        self.track_gs('get_inverses_direct after', dict_to_use=self.GSINV)

    def get_damping_factor(self):
        items = list(self.GS.items())
        for item_no, item in enumerate(items):
            if item_no%2 == 0:
                self.damping[item[0]] = torch.sqrt(torch.trace(items[item_no][1])/torch.trace(items[item_no+1][1]))
            else:
                self.damping[item[0]] = torch.sqrt(torch.trace(items[item_no][1])/torch.trace(items[item_no-1][1]))

    def get_inverses_direct_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            GSPROJINV = torch.inverse(self.GSLOWER[key] + torch.eye(self.GSLOWER[key].shape[0]) * 0.001)
            self.GSINV[key] = self.project_mtx_To_higher_space(GSPROJINV, key)


    def maintain_invs(self, params, args):
        tick = self.tick

        if args.inv_type == 'recursive' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            if True:#tick % args.inv_period == 0:
                self.get_invs_recursively()
        elif args.inv_type == 'recursive' and args.subspace_fraction < 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            if True:#tick % args.inv_period == 0:
                self.get_invs_recursively_lower()
        elif args.inv_type == 'direct' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            if tick % args.inv_period == 0:
                self.get_inverses_direct()
                print('direct inverse calculated')
        elif args.inv_type == 'direct' and args.subspace_fraction < 1:
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            if tick % args.inv_period == 0:
                self.get_inverses_direct_lower()
        else:
            raise Exception('unknown combination')
        self.tick = self.tick + 1