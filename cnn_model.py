class CNN(nn.Module):
    def __init__(self, subspace_fraction=0.1):
        super(CNN, self).__init__()
        self.subspace_fraction = subspace_fraction
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)
        self.conv_layers = [self.conv1, self.conv2, self.fc1, self.fc2]
        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = np.eye((1*3*3))
        self.GS['GAM0_AVG'] = np.eye((32))
        self.GS['PSI1_AVG'] = np.eye((32*3*3))
        self.GS['GAM1_AVG'] = np.eye((64))
        self.GS['PSI2_AVG'] = np.eye((1600))
        self.GS['GAM2_AVG'] = np.eye((128))
        self.GS['PSI3_AVG'] = np.eye((128))
        self.GS['GAM3_AVG'] = np.eye((10))
        #self.GS['PSI4_AVG'] = np.eye((128))
        #self.GS['GAM4_AVG'] = np.eye((10))

        self.GSLOWER = {}
        self.GSLOWERINV = {}
        for key, val in self.GS.items():
            self.GSLOWER[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))
            self.GSLOWERINV[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))
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
        self.a0 = None
        self.s0 = None
        self.a1 = None
        self.s1 = None
        self.s2 = None
        self.a2 = None
        self.s3 = None
        self.a3 = None
        self.s4 = None

    def get_subspace_size(self, full_space_size):
        subspace_size = int(full_space_size * self.subspace_fraction)
        if subspace_size < 64:
            subspace_size = full_space_size
        return subspace_size

    def forward(self, x):
        self.a0 = x
        #print('a0 shape = {}'.format(self.a0.shape))
        self.s0 = self.conv1(self.a0)
        #print('s0 shape = {}'.format(self.s0.shape))
        self.a1 = F.relu(self.s0)
        #print('a1 shape = {}'.format(self.a1.shape))
        x = F.max_pool2d(self.a1, 2)
        self.s1 = self.conv2(x)
        #print('s1 shape = {}'.format(self.s1.shape))
        x = F.relu(self.s1)

        x = F.max_pool2d(x, 2)
        #print('max_pool2d output x shape = {}'.format(x.shape))
        x = self.dropout1(x)
        self.a2 = torch.flatten(x, 1)
        #print('flatten output x shape = {}'.format(self.a2.shape))
        self.s2 = self.fc1(self.a2)
        #print('fc1 output x shape = {}'.format(self.s2.shape))
        self.a3 = F.relu(self.s2)
        #print('relu output x shape = {}'.format(self.a3.shape))
        x = self.dropout2(self.a3)
        self.s3 = self.fc2(x)
        #print('fc2 output x shape = {}'.format(self.s3.shape))
        output = F.log_softmax(self.s3, dim=1)
        #print('output x shape = {}'.format(output.shape))
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        self.s3.retain_grad()

        return output


    def get_grads(self):
        a0 = self.a0.detach().numpy()
        s0_grad = self.s0.grad.detach().numpy()
        a1 = self.a1.detach().numpy()
        s1_grad = self.s1.grad.detach().numpy()
        a2 = self.a2.detach().numpy()
        s2_grad = self.s2.grad.detach().numpy()
        a3 = self.a3.detach().numpy()
        s3_grad = self.s3.grad.detach().numpy()
        # print('a0.shape = {}, so_grad.shape = {}, a1.shape = {}, s1_grad.shape = {}, a2.shape = {}, s2_grad.shape = {}'.format(a0.shape, s0_grad.shape, a1.shape, s1_grad.shape, a2.shape, s2_grad.shape))
        params = [a0, s0_grad, a1, s1_grad, a2, s2_grad, a3, s3_grad]
        param_out = []
        for itr, param in enumerate(params):
            if len(param.shape) == 4 :
                #print('itr={}: in param_shape = {}'.format(itr, param.shape))
                kernel_size = self.conv_layers[itr//2].kernel_size
                padding = self.conv_layers[itr//2].padding
                stride = self.conv_layers[itr//2].stride
                if itr % 2 == 0:
                    reduced_param = im2col_indices(param, kernel_size[0], kernel_size[1], padding[0], stride[0])
                    reduced_param = reduced_param.T
                    reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param, axis=0, keepdims=True)
                else:
                    sz = param.shape
                    reduced_param = np.transpose(param,(1,0,2,3))
                    reduced_param = np.reshape(reduced_param, (sz[1],-1))
                    reduced_param = reduced_param.T
                    reduced_param = 1/float(reduced_param.shape[0]) * np.sum(reduced_param,axis=0, keepdims=True)
            elif len(param.shape) == 2:
                reduced_param = param
            else:
                raise Exception('invalid param length = {}'.format(len(param)))
            param_out.append(reduced_param)
            #print('out param_shape = {}'.format(reduced_param.shape))
        return tuple(param_out)

    def projection_matrix_update(self):
        if self.subspace_fraction == 1:
            return
        for item_no, (key, item) in enumerate(self.GS.items()):
            eigval, eigvec = np.linalg.eigh(self.GS[key])
            subspace_size = self.get_subspace_size(eigvec.shape[0])
            eigvec_subspace = eigvec[:, -subspace_size:]
            self.P[key] = eigvec_subspace

    def project_vec_to_lower_space(self, matrix, key):
        #print('project_to_lower_space: Shape of P[{}] = {}. Shape of matrix = {}'.format(key, self.P[key].shape, matrix.shape))
        if self.subspace_fraction == 1:
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
        # print('X.shape = {}, GSINV[{}].shape = {}'.format(X.shape, key, GS.shape))
        print('num_batches = {}'.format(num_batches))
        inner_term = np.eye(num_batches) + X @ GS @ X.T
        xg = X @ GS
        gx = GS @ X.T
        GS = GS - gx @ np.linalg.inv(inner_term) @ xg
        # print(
        #    'X.shape = {}, GSINV[{}].shape = {}, inner_term.shape = {}'.format(X.shape, key, self.GSLOWERINV[key].shape,
        #                                                                       inner_term.shape))
        return GS

    def maintain_corr(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            print('Overflow location {}: param[{}].shape = {}'.format(key, item_no, params[item_no].shape))
            print(params[item_no])
            self.corr_curr[item_no] = params[item_no].T @ params[item_no]

    def maintain_corr_lower(self, params):
        for item_no, (key, item) in enumerate(self.GS.items()):
            self.corr_curr_lower_proj[item_no] = self.project_vec_to_lower_space(params[item_no], key)
            self.corr_curr_lower[item_no] = self.corr_curr_lower_proj[item_no].T @ self.corr_curr_lower_proj[item_no]

    def maintain_avgs(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, self.corr_curr[item_no].shape, key, self.GS[key].shape))
            self.GS[key] = alpha * self.GS[key] + (1 - alpha) * self.corr_curr[item_no]

    def maintain_avgs_lower(self):
        alpha = 0.95
        for item_no, (key, item) in enumerate(self.GS.items()):
            #print('corr_curr[{}].shape = {}, GS.shape[{}] = {}'.format(item_no, self.corr_curr[item_no].shape, key, self.GS[key].shape))
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
            self.GSINV[key] = np.linalg.inv(self.GS[key])   + (np.eye(self.GS[key].shape[0]) * 0.001)

    def get_inverses_direct_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            GSPROJINV = np.linalg.inv(self.GSLOWER[key])  # + np.eye(GSPROJ.shape[0]) * 0.001)
            self.GSINV[key] = self.project_mtx_To_higher_space(GSPROJINV, key)

    def maintain_invs(self, params, args):
        tick = self.tick

        if args.inv_type == 'recursive' and args.subspace_fraction == 1:
            self.maintain_corr(params)
            self.maintain_avgs()
            if True:  # tick % args.inv_period == 0:
                self.get_invs_recursively()
        elif args.inv_type == 'recursive' and args.subspace_fraction < 1:
            self.maintain_corr_lower(params)
            self.maintain_avgs_lower()
            if True:  # tick % args.inv_period == 0:
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



class MLP(nn.Module):
    def __init__(self, subspace_fraction=0.1):
        super(MLP, self).__init__()
        self.subspace_fraction = subspace_fraction
        self.linear1 = nn.Linear(784, 250)
        self.linear2 = nn.Linear(250, 100)
        self.linear3 = nn.Linear(100, 10)

        self.GS = OrderedDict()
        self.GS['PSI0_AVG'] = np.eye((784))
        self.GS['GAM0_AVG'] = np.eye((250))
        self.GS['PSI1_AVG'] = np.eye((250))
        self.GS['GAM1_AVG'] = np.eye((100))
        self.GS['PSI2_AVG'] = np.eye((100))
        self.GS['GAM2_AVG'] = np.eye((10))

        self.GSLOWER = {}
        self.GSLOWERINV = {}
        for key, val in self.GS.items():
            self.GSLOWER[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))
            self.GSLOWERINV[key] = np.eye(self.get_subspace_size(self.GS[key].shape[0]))

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
        self.a0 = X
        self.s0 = self.linear1(self.a0)
        self.a1 = F.relu(self.s0)
        self.s1 = self.linear2(self.a1)
        self.a2 = F.relu(self.s1)
        self.s2 = self.linear3(self.a2)
        self.s0.retain_grad()
        self.s1.retain_grad()
        self.s2.retain_grad()
        return F.log_softmax(self.s2, dim=1)

    def get_subspace_size(self, full_space_size):
        subspace_size = int(full_space_size * self.subspace_fraction)
        if subspace_size < 64:
            subspace_size = full_space_size
        return subspace_size

    def get_grads(self):
        a0 = self.a0.detach().numpy()
        s0_grad = self.s0.grad.detach().numpy()
        a1 = self.a1.detach().numpy()
        s1_grad = self.s1.grad.detach().numpy()
        a2 = self.a2.detach().numpy()
        s2_grad = self.s2.grad.detach().numpy()
        #print('a0.shape = {}, so_grad.shape = {}, a1.shape = {}, s1_grad.shape = {}, a2.shape = {}, s2_grad.shape = {}'.format(a0.shape, s0_grad.shape, a1.shape, s1_grad.shape, a2.shape, s2_grad.shape))
        return (a0, s0_grad, a1, s1_grad, a2, s2_grad)

    def projection_matrix_update(self):
        if self.subspace_fraction == 1:
            return
        for item_no, (key, item) in enumerate(self.GS.items()):
            eigval, eigvec = np.linalg.eigh(self.GS[key])
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
        inner_term = np.eye(num_batches) + X @ GS @ X.T
        xg = X @ GS
        gx = GS @ X.T
        GS = GS - gx @ np.linalg.inv(inner_term) @ xg
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
            self.GSINV[key] = np.linalg.inv(self.GS[key])# + np.eye(GSPROJ.shape[0]) * 0.001)


    def get_inverses_direct_lower(self):
        for item_no, (key, item) in enumerate(self.GS.items()):
            GSPROJINV = np.linalg.inv(self.GSLOWER[key])# + np.eye(GSPROJ.shape[0]) * 0.001)
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