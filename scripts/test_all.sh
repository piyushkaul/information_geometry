#./scripts/test_grid_search_cnn_direct.sh
#python main.py --model resnet18 --gamma 0.8 --lr 0.001 --optimizer adam_ngd --subspace-fraction 1 --inv-period 50 --proj-period 50 --inv-type direct --dataset cifar10 --epochs 100
python main.py --model resnet18 --gamma 0.8 --lr 0.01 --optimizer adam_ngd --subspace-fraction 1 --inv-period 50 --proj-period 50 --inv-type direct --dataset cifar10 --epochs 100
python main.py --model resnet18 --gamma 0.8 --lr 0.01 --optimizer adam_ngd --subspace-fraction 1 --inv-period 1 --proj-period 1 --inv-type direct --dataset cifar10 --epochs 100
python main.py --model resnet18 --gamma 0.8 --lr 0.01 --optimizer adam_ngd --subspace-fraction 1 --inv-period 500 --proj-period 500 --inv-type recursive --dataset cifar10 --epochs 100
python main.py --model resnet18 --gamma 0.8 --lr 0.01 --optimizer adam_ngd --subspace-fraction 0.95 --inv-period 50 --proj-period 50 --inv-type direct --dataset cifar10 --epochs 100
python main.py --model resnet18 --gamma 0.8 --lr 0.1 --optimizer sgd --dataset cifar10 --epochs 100

