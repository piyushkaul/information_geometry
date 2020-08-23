from datetime import datetime
import numpy as np

def get_file_suffix(args):
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")
    suffix = date_time + '_' + args.optimizer + '_lr_' + str(args.lr) + '_gamma_' + str(args.gamma) + '_frac_' + \
             str(args.subspace_fraction) + '_' + args.dataset + '_' + args.inv_type + '_inv_period_' + str(args.inv_period) \
             + '_proj_period_' + str(args.proj_period) + '_model_' + str(args.model) + '_epochs_' + str(args.epochs)
    return suffix

def save_files(loss_list, tag, suffix):
    loss_list_np = np.array(loss_list)
    loss_filename = 'temp/' + tag + suffix + 'txt'
    loss_list_np.tofile(loss_filename, '\n', '%f')

