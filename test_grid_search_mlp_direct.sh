for opt in ngd sgd
do
   for gamma in 0.7 0.5 0.3 0.1
   do
        for lr in 0.5 0.1 0.05 0.01 0.005 0.001
        do
             for subspace_fraction in  1
             do
                   for inv_period in 50
                   do
                         for proj_period in 50
                         do
                               for inv_type in direct 
                               do
                                     echo python mnist.py --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type  --dataset fashion_mnist --epochs 15
                                     python mnist.py --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type --dataset fashion_mnist --epochs 15
                              done
                         done
                    done
             done          
        done
   done                                                                                                                                                                                                                          
done
