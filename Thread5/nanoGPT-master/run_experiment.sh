#!/bin/bash

# 指定要遍历的dropout值
dropout_values=(0 0.2 0.5)

for dropout in "${dropout_values[@]}"
do
    echo "Running training with dropout=$dropout"
    python train_test.py --dropout=$dropout
done

