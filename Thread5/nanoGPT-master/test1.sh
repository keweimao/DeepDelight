#!/bin/bash

# In this test res dropout are set to 0 and I try to find the influence on attn dropout.
# --------------------------------------------------------------------------


# Set the output file name
output_file="training_times_test1.txt"

# Clear the existing content of the output file (if you do not wish to keep previous records)
> "$output_file"

# Record the start time
start_time=$(date +%s)

# Train the model
python train_a0r0.py config/train_shakespeare_char_a0.py

# Record the end time
end_time=$(date +%s)

# Calculate and display the training time
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a0r0: $min min $sec s" >> "$output_file"

# Repeat the above steps for other training scripts

# For train_a2r0.py
start_time=$(date +%s)
python train_a2r0.py config/train_shakespeare_char_a2.py
end_time=$(date +%s)
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a2r0: $min min $sec s" >> "$output_file"

# For train_a5r0.py
start_time=$(date +%s)
python train_a5r0.py config/train_shakespeare_char_a5.py
end_time=$(date +%s)
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a5r0: $min min $sec s" >> "$output_file"

# Run the visualization script
python visualization_a.py
