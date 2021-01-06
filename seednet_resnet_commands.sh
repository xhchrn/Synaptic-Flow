python3 main.py --gpu 1 \
    --dataset cifar10 --model-class seednet --model resnet18 --init-method standard \
    --train-batch-size 128 --post-epochs 1 \
    --pruner snip --compression 1.0 \
    --experiment singleshot --expid 0 \
    --result-dir seednet/resnet18/standard_snip_sp0.1 --seed 118

python3 main.py --gpu 2 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/lenet/initial_classifier_state_dict_lt.pt \
    --pruner grasp --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_grasp --seed 118

python3 main.py --gpu 3 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/lenet/initial_classifier_state_dict_lt.pt \
    --pruner synflow --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_snip --seed 118

python3 main.py --gpu 4 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/lenet/initial_classifier_state_dict_lt.pt \
    --pruner mag --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_mag --seed 118


# ==============================================================================

python3 main.py --gpu 1 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.96910013 \
    --classifier-compression 0.45757491 \
    --mask-scope local \
    --optimizer sgd --lr 0.1 --weight-decay 0.0 \
    --pruner synflow \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_synflow --seed 118

python3 main.py --gpu 0 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.96910013 \
    --classifier-compression 0.45757491 \
    --mask-scope local \
    --optimizer sgd --lr 0.1 --weight-decay 0.0 \
    --pruner grasp \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_grasp --seed 118

python3 main.py --gpu 0 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.96910013 \
    --classifier-compression 0.45757491 \
    --mask-scope local \
    --optimizer sgd --lr 0.1 --weight-decay 0.0 \
    --pruner snip \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_snip --seed 118

python3 main.py --gpu 0 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.96910013 \
    --classifier-compression 0.45757491 \
    --mask-scope local \
    --optimizer sgd --lr 0.1 --weight-decay 0.0 \
    --pruner mag \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_mag --seed 118


for prune_method in synflow grasp snip mag; do
    for seed in 118 218 318 418 518; do
        python3 main.py --gpu 1 \
            --dataset mnist --model lenet-300-100 \
            --train-batch-size 128 --post-epochs 20 \
            --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
            --compression 0.96910013 \
            --classifier-compression 0.45757491 \
            --mask-scope local \
            --optimizer sgd --lr 0.1 --weight-decay 0.0 \
            --pruner $prune_method \
            --experiment singleshot --expid 0 \
            --result-dir "mnist/lenet_${prune_method}_${seed}" --seed $seed
    done
done



for prune_method in synflow grasp snip mag; do
    for seed in 118 218 318 418 518; do
        python3 main.py --gpu 0 \
            --dataset mnist --model lenet-300-100 \
            --train-batch-size 128 --post-epochs 5 \
            --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
            --compression 0.96910013 \
            --classifier-compression 0.45757491 \
            --pre-steps 20 \
            --mask-scope local \
            --optimizer sgd --lr 0.1 --weight-decay 0.0 \
            --pruner $prune_method \
            --experiment singleshot --expid 0 \
            --result-dir "mnist/lenet_${prune_method}_pre20it_${seed}" --seed $seed
    done
done

for prune_method in synflow grasp snip mag; do
    for seed in 118 218 318 418 518; do
        python3 main.py --gpu 0 \
            --dataset mnist --model lenet-300-100 \
            --train-batch-size 128 --post-epochs 5 \
            --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
            --compression 0.96910013 \
            --classifier-compression 0.45757491 \
            --pre-steps 100 \
            --mask-scope local \
            --optimizer sgd --lr 0.1 --weight-decay 0.0 \
            --pruner $prune_method \
            --experiment singleshot --expid 0 \
            --result-dir "mnist/lenet_${prune_method}_pre100it_${seed}" --seed $seed
    done
done
