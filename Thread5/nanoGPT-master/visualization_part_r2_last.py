import json
import matplotlib.pyplot as plt
import numpy as np


with open('val_losses_a2r0_2.json', 'r') as file:
    val_losses_a2r0_2 = json.load(file)

with open('val_losses_a2r05_2.json', 'r') as file:
    val_losses_a2r05_2 = json.load(file)


with open('val_losses_a2r25_2.json', 'r') as file:
    val_losses_a2r25_2 = json.load(file)



with open('val_losses_a2r5_2.json', 'r') as file:
    val_losses_a2r5_2 = json.load(file)


# with open('val_losses_a2r75_2.json', 'r') as file:
#     val_losses_a2r75_2 = json.load(file)

with open('val_losses_a2r100_2.json', 'r') as file:
    val_losses_a2r100_2 = json.load(file)



n = len(val_losses_a2r0_2)
x_values = np.linspace(0, 50000, n)


# # # Plotting
# plt.figure(figsize=(10, 6))
# # # plt.plot(train_losses_a2r0, label='Training Loss a2r0')
    
# plt.plot(x_values, val_losses_a2r0_2, label='Validation Loss a2r0_2')

# plt.plot(x_values, val_losses_a2r05_2, label='Validation Loss a2r05_2')

# # plt.plot(val_losses_a2r1_2, label='Validation Loss a2r1_2')

# plt.plot(x_values, val_losses_a2r25_2, label='Validation Loss a2r25_2')

# plt.plot(x_values, val_losses_a2r5_2, label='Validation Loss a2r5_2')

# # plt.plot(val_losses_a2r75_2, label='Validation Loss a2r75_2')

# plt.plot(x_values, val_losses_a2r100_2, label='Validation Loss a2r100_2')

# # plt.plot(val_losses_a2r10, label='Validation Loss a2r10')

# 初始化图形
n = len(val_losses_a2r0_2)  # 假设所有数组长度相同，否则需要对每个数组单独处理
x_values = np.linspace(0, 50000, n)

# 初始化图形
plt.figure(figsize=(10, 6))

# 绘制后半部分的验证损失
for val_losses, label in zip([val_losses_a2r0_2, val_losses_a2r05_2, val_losses_a2r25_2, val_losses_a2r5_2, val_losses_a2r100_2], 
                             ['a=0,r1=0.025,r2=0', 'a=0,r1=0.025,r2=0.05', 'a=0,r1=0.025,r2=0.25', 'a=0,r1=0.025,r2=0.5','a=0,r1=0.025,r2=0.99']):
    start_index = len(val_losses) // 2  # 计算后半部分的起始索引
    plt.plot(x_values[start_index:], val_losses[start_index:], label=label)

plt.title('Validation Loss over Iterations (Second Half)')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)


plt.savefig('loss_plot_part_r2_last.png')