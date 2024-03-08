import json
import matplotlib.pyplot as plt
import numpy as np


with open('val_losses_a2r0_2.json', 'r') as file:
    val_losses_a2r0_2 = json.load(file)

with open('val_losses_a2r05_2.json', 'r') as file:
    val_losses_a2r05_2 = json.load(file)


# with open('val_losses_a2r1_2.json', 'r') as file:
#     val_losses_a2r1_2 = json.load(file)

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
plt.figure(figsize=(10, 6))
# # plt.plot(train_losses_a2r0, label='Training Loss a2r0')
    
plt.plot(x_values, val_losses_a2r0_2, label='a=0,r1=0.025,r2=0')

plt.plot(x_values, val_losses_a2r05_2, label='a=0,r1=0.025,r2=0.05')

# plt.plot(val_losses_a2r1_2, label='Validation Loss a2r1_2')

plt.plot(x_values, val_losses_a2r25_2, label='a=0,r1=0.025,r2=0.25')

plt.plot(x_values, val_losses_a2r5_2, label='a=0,r1=0.025,r2=0.5')

# plt.plot(val_losses_a2r75_2, label='Validation Loss a2r75_2')

plt.plot(x_values, val_losses_a2r100_2, label='a=0,r1=0.025,r2=0.99')

# plt.plot(val_losses_a2r10, label='Validation Loss a2r10')

plt.title('Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot_r_2.png')
# plt.savefig('test_loss_plot.png')


