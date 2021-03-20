for opt in adam_ngd sgd_ngd
do
   for gamma in  0.8 0.7
   do
       for lr in  0.001
        do
             for subspace_fraction in  1  0.95
             do
                   for inv_period in 50 
                   do
                         for proj_period in 50
                         do
                               for inv_type in direct recursive
                               do
                                     echo python main.py --model cnn --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type  --dataset fashion_mnist --epochs 15
                                     python main.py --model cnn --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type --dataset fashion_mnist --epochs 15
                              done
                         done
                    done
             done          
        done
   done                                                                                                                                                                                                                          
done

for opt in sgd adam
do
   for gamma in  0.8 0.7 
   do
        for lr in  0.1  0.01 0.001
        do
                                     echo python main.py --model cnn  --gamma $gamma --lr $lr --optimizer $opt  --dataset fashion_mnist --epochs 15
                                     python main.py --model cnn --gamma $gamma --lr $lr --optimizer $opt --dataset fashion_mnist --epochs 100
        done
   done                                                                                                                                                                                                                          
done
