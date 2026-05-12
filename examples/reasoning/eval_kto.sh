#!/bin/bash
# eval_kto.sh
# Minimal script to demonstrate KTO algorithm progress.

echo "Starting KTO Evaluation with a small model..."
# Using a very small model (Pythia 14M) and running for just 5 steps to show the loss updates

python -m thinkrl.cli.main kto \
    --model EleutherAI/pythia-14m \
    --dataset trl-internal-testing/math-prompt-examples \
    --dataset-split train \
    --prompt-column prompt \
    --target-column target \
    --beta 0.1 \
    --learning-rate 5e-5 \
    --batch-size 2 \
    --steps 5 \
    --no-bf16 \
    --no-flash-attention \
    --output-dir output/kto_eval
