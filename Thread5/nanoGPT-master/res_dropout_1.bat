@echo off

python train_a2r0.py config/train_shakespeare_char_a2.py
python train_a2r2.py config/train_shakespeare_char_a2.py
python train_a2r5.py config/train_shakespeare_char_a2.py
python train_a2r10.py config/train_shakespeare_char_a2.py

python visualization_r.py