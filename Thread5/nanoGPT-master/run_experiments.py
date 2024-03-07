import subprocess

# 定义要测试的dropout值列表
dropout_values = [0, 0.2, 0.5]

for dropout in dropout_values:
    print(f"Running training with dropout={dropout}")
    # 调用train.py并传入dropout值
    subprocess.run(['python', 'train_test.py', '--dropout', str(dropout)])
