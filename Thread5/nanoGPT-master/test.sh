#!/bin/bash

# Record the start time
start_time=$(date +%s)

# Train a model using Shakespeare character data
python train.py config/train_shakespeare_char.py
# python sample.py --out_dir=out-shakespeare-char


# Record the end time
end_time=$(date +%s)

# Calculate and display the training time
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time is $min min $sec s"
