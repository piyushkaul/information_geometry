import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="darkgrid")

dirname='temp/'
prefixes=['elapsed_time', 'test_accuracy', 'train_accuracy', 'train_loss', 'test_loss']
suffix='2021_03_11_01_32_01_adam_ngd_lr_0.1_gamma_0.9_frac_1.0_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt'
suffix='2021_03_11_14_45_57_adam_ngd_lr_0.01_gamma_0.9_frac_1.0_fashion_mnist_direct_inv_period_500_proj_period_50_model_mlp_epochs_15_batch_size_64txt'

suffixes=['2021_03_12_01_40_24_adam_lr_0.001_gamma_0.8_frac_1_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt',
          #'2021_03_11_18_10_03_adam_ngd_lr_0.001_gamma_0.8_frac_1.0_fashion_mnist_recursive_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt',
          '2021_03_12_10_50_58_adam_ngd_lr_0.001_gamma_0.8_frac_1.0_fashion_mnist_direct_inv_period_1_proj_period_1_model_mlp_epochs_15_batch_size_64txt',
          '2021_03_12_11_53_10_adam_ngd_lr_0.001_gamma_0.8_frac_1.0_fashion_mnist_recursive_inv_period_500_proj_period_500_model_mlp_epochs_15_batch_size_64txt',
           '2021_03_12_19_30_26_adam_ngd_lr_0.001_gamma_0.8_frac_0.95_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt',
          '2021_03_12_20_06_20_sgd_lr_0.1_gamma_0.8_frac_1.0_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_15_batch_size_64txt'
          ]

params = {}
for suffix in suffixes:
    params[suffix] = {}

for suffix in suffixes:
    for prefix in prefixes:
        filename = dirname + prefix + suffix
        with open(filename, 'r') as fp:
            params[suffix][prefix] = np.fromfile(fp, dtype=np.float, sep='\n')
            #print(prefix)
            #print(params)

for suffix in suffixes:
    for idx, (params_name, val) in enumerate(params[suffix].items()):
        if idx==0:
            continue
        plt.subplot(511+int(idx))
        plt.plot(params[suffix]['elapsed_time'][1:], params[suffix][params_name], 'r', label=params_name)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc=2, fontsize="small")
plt.show()