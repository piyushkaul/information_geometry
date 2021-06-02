#test_accuracy2021_03_12_18_53_13_adam_ngd_lr_0.001_gamma_0.8_frac_1.0_fashion_mnist_direct_inv_period_1_proj_period_1_model_mlp_epochs_100_batch_size_64txt
#test_accuracy2021_03_12_19_29_19_adam_ngd_lr_0.001_gamma_0.8_frac_1.0_fashion_mnist_recursive_inv_period_500_proj_period_500_model_mlp_epochs_100_batch_size_64txt
#test_accuracy2021_03_12_19_56_01_adam_ngd_lr_0.001_gamma_0.8_frac_0.95_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_100_batch_size_64txt
#test_accuracy2021_03_12_20_15_50_sgd_lr_0.1_gamma_0.8_frac_0.95_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_100_batch_size_64txt
#test_accuracy2021_03_12_20_33_40_adam_lr_0.1_gamma_0.8_frac_1.0_fashion_mnist_direct_inv_period_50_proj_period_50_model_mlp_epochs_100_batch_size_64txt

#for i in {1..30}; 
#do
#python main.py --model mlp --gamma 0.8 --lr 0.001 --optimizer adam_ngd --subspace-fraction 1 --inv-period 50 --proj-period 50 --inv-type direct --dataset fashion_mnist --epochs 100 --seed $i
#done
for i in {1..30};
do
python main.py --model mlp --gamma 0.8 --lr 0.001 --optimizer adam_ngd --subspace-fraction 0.95 --inv-period 50 --proj-period 50 --inv-type recursive --dataset fashion_mnist --epochs 100 --seed $i
done
for i in {1..30};
do
python main.py --model mlp --gamma 0.8 --lr 0.001 --optimizer sgd --dataset fashion_mnist --epochs 100 --seed $i
done
for i in {1..30}; 
do
python main.py --model mlp --gamma 0.8 --lr 0.001 --optimizer adam_ngd --subspace-fraction 0.95 --inv-period 50 --proj-period 50 --inv-type recursive --dataset fashion_mnist --epochs 100 --random-projection --seed $i
done
