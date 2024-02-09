import json
import matplotlib.pyplot as plt
import numpy as np

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



# with open('train_losses_a2r1.json', 'r') as file:
#     train_losses_a2r1 = json.load(file)

# with open('val_losses_a2r1.json', 'r') as file:
#     val_losses_a2r1 = json.load(file)

# with open('train_losses_a2r2.json', 'r') as file:
#     train_losses_a2r2 = json.load(file)

# with open('val_losses_a2r2.json', 'r') as file:
#     val_losses_a2r2 = json.load(file)

# with open('train_losses_origin.json', 'r') as file:
#     train_losses_origin = json.load(file)

# with open('val_losses_origin.json', 'r') as file:
#     val_losses_origin = json.load(file)

n = len(train_losses_a0r0)
x_values = np.linspace(0, 500000, n)
# # Plotting
plt.figure(figsize=(10, 6))
# plt.plot(train_losses_a2r0, label='Training Loss a2r0')
plt.plot(x_values, val_losses_a0r0, label='a=0,r1=0')

# plt.plot(train_losses_a1r0, label='Training Loss a2r0')
plt.plot(x_values, val_losses_a2r0, label='a=0.2,r1=0')

# plt.plot(train_losses_a2r1, label='Training Loss a2r1')
plt.plot(x_values, val_losses_a5r0, label='a=0.5,r1=0')


# plt.plot(train_losses_origin, label='Training Loss Origin')
# plt.plot(val_losses_origin, label='Validation Loss Origin')

plt.title('Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Validation Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot_a.png')
# plt.savefig('test_loss_plot.png')


