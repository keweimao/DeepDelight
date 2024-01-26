@echo off

python train_a0r0.py config/train_shakespeare_char_a0.py
python train_a2r0.py config/train_shakespeare_char_a2.py
python train_a5r0.py config/train_shakespeare_char_a5.py

python visualization_a.py


