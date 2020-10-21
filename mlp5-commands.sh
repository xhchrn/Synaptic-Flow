python3 main.py --gpu 1 \
    --dataset mnist --model fc \
    --train-batch-size 128 \
    --pruner synflow --compression 1.0 \
    --experiment singleshot \
    --result-dir mlp5_sf --seed 118
