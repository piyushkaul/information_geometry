import torch.nn as nn
from collections import OrderedDict
import torch

class ModelFIM(nn.Module):
    def __init__(self, subspace_fraction=0.1):
        super(ModelFIM, self).__init__()
        self.subspace_fraction = subspace_fraction
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = torch.eye((784))
        self.GS['GAM0_AVG'] = torch.eye((250))
        self.GS['PSI1_AVG'] = torch.eye((250))
        self.GS['GAM1_AVG'] = torch.eye((100))
        self.GS['PSI2_AVG'] = torch.eye((100))
        self.GS['GAM2_AVG'] = torch.eye((10))

        self.GSLOWER = {}
        self.GSLOWERINV = {}
        for key, val in self.GS.items():
            self.GSLOWER[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0]))
            self.GSLOWERINV[key] = torch.eye(self.get_subspace_size(self.GS[key].shape[0]))

        self.GSINV = {}

        self.P = {}
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = self.GS[key]
            subspace_size = self.get_subspace_size(self.GSINV[key].shape[0])
            eigvec_subspace = self.GS[key][:, -subspace_size:]
            self.P[key] = eigvec_subspace

        self.corr_curr = [None]*len(self.GS)
        self.corr_curr_lower_proj = [None] * len(self.GS)
        self.corr_curr_lower = [None] * len(self.GS)
        self.tick = 0


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
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.corr_curr[item_no] = params[item_no].T @ params[item_no]

    def maintain_corr_lower(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.corr_curr_lower_proj[item_no] = self.project_vec_to_lower_space(params[item_no], key)
            self.corr_curr_lower[item_no] = self.corr_curr_lower_proj[item_no].T @ self.corr_curr_lower_proj[item_no]

    def maintain_avgs(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GS[key] = alpha * self.GS[key] + (1 - alpha) * self.corr_curr[item_no]

    def maintain_avgs_lower(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GSLOWER[key] = alpha * self.GSLOWER[key] + (1 - alpha) * self.GSLOWER[key]


    def get_invs_recursively(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            XFULL = self.corr_curr[item_no]
            self.GSINV[key] = self.matrix_inv_lemma(XFULL, self.GSINV[key], key=key)

    def get_invs_recursively_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            XLOWER = self.corr_curr_lower_proj[item_no]
            self.GSLOWERINV[key] = self.matrix_inv_lemma(XLOWER, self.GSLOWERINV[key], key=key)


    def get_inverses_direct(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.GSINV[key] = torch.inverse(self.GS[key]) + (torch.eye(self.GS[key].shape[0]) * 0.001)


    def get_inverses_direct_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            GSPROJINV = torch.inverse(self.GSLOWER[key])# + np.eye(GSPROJ.shape[0]) * 0.001)
            self.GSINV[key] = self.project_mtx_To_higher_space(GSPROJINV, key)


    def maintain_invs(self, params, args):
        tick = self.tick

        if args.inv_type == 'recursive' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            if True:#tick % args.inv_period == 0:
                self.get_invs_recursively()
        elif args.inv_type == 'recursive' and args.subspace_fraction < 1:
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            if True:#tick % args.inv_period == 0:
                self.get_invs_recursively_lower()
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
        self.tick = self.tick + 1