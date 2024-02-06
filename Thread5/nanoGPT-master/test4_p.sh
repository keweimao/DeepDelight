#!/bin/bash

# In this test attn dropout are set to 0 and I try to find the influence on one layer res dropout.
# --------------------------------------------------------------------------


# Set the output file name
output_file="training_times_test4_p.txt"

# Clear the existing content of the output file (if you do not wish to keep previous records)
> "$output_file"

# Record the start time
start_time=$(date +%s)

# Train the model
python train_a2r0_4_p.py config/train_shakespeare_char_a2.py

# Record the end time
end_time=$(date +%s)

# Calculate and display the training time
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a2r0_4_p: $min min $sec s" >> "$output_file"

# Repeat the above steps for other training scripts

# For train_a2r2.py
start_time=$(date +%s)
python train_a2r2_4_p.py config/train_shakespeare_char_a2.py
end_time=$(date +%s)
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a2r2_4_p: $min min $sec s" >> "$output_file"

# For train_a2r5.py
start_time=$(date +%s)
python train_a2r5_4_p.py config/train_shakespeare_char_a2.py
end_time=$(date +%s)
cost_time=$(( end_time - start_time ))
min=$(( cost_time / 60 ))
sec=$(( cost_time % 60 ))
echo "Training time for a2r5_4_p: $min min $sec s" >> "$output_file"

# Run the visualization script
python visualization_r4_p.py
