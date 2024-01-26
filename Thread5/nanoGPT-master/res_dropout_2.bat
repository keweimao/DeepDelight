@echo off

python train_a2r0_2.py config/train_shakespeare_char_a2.py
python train_a2r2_2.py config/train_shakespeare_char_a2.py
python train_a2r5_2.py config/train_shakespeare_char_a2.py

python visualization_r2.py