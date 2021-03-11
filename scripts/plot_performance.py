import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")

dirname='temp/'
prefixes=['elapsed_time', 'test_accuracy', 'train_accuracy', 'train_loss', 'test_loss']
suffix='2021_03_11_01_32_01_adam_ngd_lr_0.1_gamma_0.9_frac_1.0_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt'
suffix='2021_03_11_14_45_57_adam_ngd_lr_0.01_gamma_0.9_frac_1.0_fashion_mnist_direct_inv_period_500_proj_period_50_model_mlp_epochs_15_batch_size_64txt'
params = {}
for prefix in prefixes:
    filename = dirname + prefix + suffix
    with open(filename, 'r') as fp:
        params[prefix] = np.fromfile(fp, dtype=np.float, sep='\n')
        #print(prefix)
        #print(params)

for idx, (params_name, val) in enumerate(params.items()):
    if idx==0:
        continue
    plt.subplot(511+int(idx))
    plt.plot(range(len(params[params_name])), params[params_name], 'r', label=params_name)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc=2, fontsize="small")
plt.show()