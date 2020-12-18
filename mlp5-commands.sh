python3 main.py --gpu 1 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/lenet/initial_classifier_state_dict_lt.pt \
    --pruner synflow --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_sf --seed 118

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
    --pruner snip --compression 1.0 --mask-scope local \
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
    --compression 0.8721901170725075 --mask-scope local \
    --pruner synflow \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_synflow --seed 118

python3 main.py --gpu 2 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.8721901170725075 --mask-scope local \
    --pruner grasp \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_grasp --seed 118

python3 main.py --gpu 3 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.8721901170725075 --mask-scope local \
    --pruner snip \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_snip --seed 118

python3 main.py --gpu 4 \
    --dataset mnist --model lenet-300-100 \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ../LTH-Pytorch/lenet_mnist_init_state_dict_for_synflow_repo.pth.tar \
    --compression 0.8721901170725075 --mask-scope local \
    --pruner mag \
    --experiment singleshot --expid 0 \
    --result-dir mnist/lenet_mag --seed 118
