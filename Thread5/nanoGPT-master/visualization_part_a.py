import json
import matplotlib.pyplot as plt
import numpy as np
# with open('train_losses_a2r0_2.json', 'r') as file:
#     train_losses_a2r0_2 = json.load(file)

with open('train_losses_a0r0.json', 'r') as file:
    train_losses_a0r0 = json.load(file)

with open('val_losses_a0r0.json', 'r') as file:
    val_losses_a0r0 = json.load(file)

with open('train_losses_a2r0.json', 'r') as file:
    train_losses_a2r0 = json.load(file)

with open('val_losses_a2r0.json', 'r') as file:
    val_losses_a2r0 = json.load(file)

with open('train_losses_a5r0.json', 'r') as file:
    train_losses_a5r0 = json.load(file)

with open('val_losses_a5r0.json', 'r') as file:
    val_losses_a5r0 = json.load(file)



# 生成整个迭代范围的x_values
n = len(train_losses_a0r0)  # 假设所有数组长度相同，否则需要对每个数组单独处理
x_values = np.linspace(0, 500000, n)

# 初始化图形
plt.figure(figsize=(10, 6))

# 绘制后半部分的验证损失
for val_losses, label in zip([val_losses_a0r0, val_losses_a2r0, val_losses_a5r0], ['a=0,r1=0', 'a=0.2,r1=0', 'a=0.5,r1=0']):
    start_index = len(val_losses) // 2  # 计算后半部分的起始索引
    plt.plot(x_values[start_index:], val_losses[start_index:], label=label)

plt.title('Validation Loss over Iterations (Second Half)')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)



plt.savefig('loss_plot_part_a.png')


