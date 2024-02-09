import json
import matplotlib.pyplot as plt
import numpy as np

with open('val_losses_a2r0.json', 'r') as file:
    val_losses_a2r0 = json.load(file)

with open('val_losses_a2r025.json', 'r') as file:
    val_losses_a2r025 = json.load(file)

with open('val_losses_a2r05.json', 'r') as file:
    val_losses_a2r05 = json.load(file)

with open('val_losses_a2r1.json', 'r') as file:
    val_losses_a2r1 = json.load(file)

with open('val_losses_a2r2.json', 'r') as file:
    val_losses_a2r2 = json.load(file)


with open('val_losses_a2r5.json', 'r') as file:
    val_losses_a2r5 = json.load(file)





val_losses_a2r0 = val_losses_a2r0[:201]


# # # Plotting
# plt.figure(figsize=(10, 6))
# # plt.plot(train_losses_a2r0, label='Training Loss a2r0')
    


n = len(val_losses_a2r025)
x_values = np.linspace(0, 50000, n)

plt.figure(figsize=(10, 6))

plt.plot(x_values, val_losses_a2r0, label='a=2, r1=0')

plt.plot(x_values, val_losses_a2r025, label='a=2, r1=0.025')

# plt.plot(val_losses_a2r05, label='Validation Loss a2r05')

# plt.plot(val_losses_a2r1, label='Validation Loss a2r1')

plt.plot(x_values, val_losses_a2r2, label='a=2, r1=0.2')

plt.plot(x_values, val_losses_a2r5, label='a=2, r1=0.5')

# plt.plot(val_losses_a2r10, label='Validation Loss a2r10')

plt.title('Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot_r.png')
# plt.savefig('test_loss_plot.png')


