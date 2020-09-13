for opt in ngd 
do
   for gamma in 0.99 0.9 0.8 0.6 
   do
        for lr in  0.1 0.05 0.01 0.005 0.001
        do
             for subspace_fraction in  1 0.5
             do
                   for inv_period in 50
                   do
                         for proj_period in 50
                         do
                               for inv_type in direct 
                               do
                                     echo python mnist.py --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type  --dataset fashion_mnist --epochs 100
                                     python mnist.py --gamma $gamma --lr $lr --optimizer $opt --subspace-fraction $subspace_fraction --inv-period $inv_period --proj-period $proj_period --inv-type $inv_type --dataset fashion_mnist --epochs 100
                              done
                         done
                    done
             done          
        done
   done                                                                                                                                                                                                                          
done

for opt in sgd 
do
   for gamma in 0.99 0.9 0.8 0.6 
   do
        for lr in  0.1 0.05 0.01 0.005 0.001
        do
                                     echo python mnist.py --gamma $gamma --lr $lr --optimizer $opt  --dataset fashion_mnist --epochs 100
                                     python mnist.py --gamma $gamma --lr $lr --optimizer $opt --dataset fashion_mnist --epochs 100
        done
   done                                                                                                                                                                                                                          
done
