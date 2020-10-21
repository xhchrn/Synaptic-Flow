python3 main.py --gpu 1 \
    --dataset mnist --model fc \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/mlp/initial_classifier_state_dict_lt.pt \
    --pruner synflow --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mlp5_sf --seed 118

python3 main.py --gpu 2 \
    --dataset mnist --model fc \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/mlp/initial_classifier_state_dict_lt.pt \
    --pruner grasp --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mlp5_grasp --seed 118

python3 main.py --gpu 3 \
    --dataset mnist --model fc \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/mlp/initial_classifier_state_dict_lt.pt \
    --pruner snip --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mlp5_snip --seed 118

python3 main.py --gpu 4 \
    --dataset mnist --model fc \
    --train-batch-size 128 --post-epochs 20 \
    --load-init ./mnist/mlp/initial_classifier_state_dict_lt.pt \
    --pruner mag --compression 1.0 --mask-scope local \
    --experiment singleshot --expid 0 \
    --result-dir mlp5_mag --seed 118
