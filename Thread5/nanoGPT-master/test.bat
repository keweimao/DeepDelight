@echo off

python train_a0r0.py config/train_shakespeare_char_a0r0.py
python train_a1r0.py config/train_shakespeare_char_a1r0.py
python train_a2r0.py config/train_shakespeare_char_a2r0.py
python train_a2r1.py config/train_shakespeare_char_a2r1.py
python train_a2r2.py config/train_shakespeare_char_a2r2.py

python visualization_a.py
python visualization_part_a.py

python visualization_r.py
python visualization_part_r.py

