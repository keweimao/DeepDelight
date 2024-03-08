import json
import matplotlib.pyplot as plt
import numpy as np
# with open('train_losses_a2r0_2.json', 'r') as file:
#     train_losses_a2r0_2 = json.load(file)

with open('val_losses_a2r0_4.json', 'r') as file:
    val_losses_a2r0_4 = json.load(file)


# with open('train_losses_a2r2_2.json', 'r') as file:
#     train_losses_a2r2_2 = json.load(file)

with open('val_losses_a2r2_4.json', 'r') as file:
    val_losses_a2r2_4 = json.load(file)

# with open('train_losses_a2r5_2.json', 'r') as file:
#     train_losses_a2r5_2 = json.load(file)

with open('val_losses_a2r5_4.json', 'r') as file:
    val_losses_a2r5_4 = json.load(file)

# with open('train_losses_a2r10.json', 'r') as file:
#     train_losses_a2r10 = json.load(file)

# with open('val_losses_a2r10.json', 'r') as file:
#     val_losses_a2r10 = json.load(file)


with open('val_losses_a2r99_4.json', 'r') as file:
    val_losses_a2r99_4 = json.load(file)



n = len(val_losses_a2r0_4)  # 假设所有数组长度相同，否则需要对每个数组单独处理
x_values = np.linspace(0, 50000, n)

# 初始化图形
plt.figure(figsize=(10, 6))

# 绘制后半部分的验证损失
for val_losses, label in zip([val_losses_a2r0_4, val_losses_a2r2_4, val_losses_a2r5_4, val_losses_a2r99_4], 
                             ['a=0,r1=0.025,r2=0.99,r4=0','a=0,r1=0.025,r2=0.99,r4=0.2','a=0,r1=0.025,r2=0.99,r4=0.5','a=0,r1=0.025,r2=0.99,r4=0.99']):
    start_index = len(val_losses) // 2  # 计算后半部分的起始索引
    plt.plot(x_values[start_index:], val_losses[start_index:], label=label)

plt.title('Validation Loss over Iterations (Second Half)')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)


plt.savefig('loss_plot_part_r4_last.png')


