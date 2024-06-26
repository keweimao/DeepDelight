import json
import matplotlib.pyplot as plt

# with open('train_losses_a0r0.json', 'r') as file:
#     train_losses_a0r0 = json.load(file)

with open('val_losses_a0r0_p.json', 'r') as file:
    val_losses_a0r0 = json.load(file)

# with open('train_losses_a2r0.json', 'r') as file:
#     train_losses_a2r0 = json.load(file)

with open('val_losses_a2r0_p.json', 'r') as file:
    val_losses_a2r0 = json.load(file)

# with open('train_losses_a5r0.json', 'r') as file:
#     train_losses_a5r0 = json.load(file)

with open('val_losses_a5r0_p.json', 'r') as file:
    val_losses_a5r0 = json.load(file)





# # Plotting
plt.figure(figsize=(10, 6))
# plt.plot(train_losses_a2r0, label='Training Loss a2r0')
plt.plot(val_losses_a0r0, label='Validation Loss a0r0')

# plt.plot(train_losses_a1r0, label='Training Loss a2r0')
plt.plot(val_losses_a2r0, label='Validation Loss a2r0')

# plt.plot(train_losses_a2r1, label='Training Loss a2r1')
plt.plot(val_losses_a5r0, label='Validation Loss a5r0')


# plt.plot(train_losses_origin, label='Training Loss Origin')
# plt.plot(val_losses_origin, label='Validation Loss Origin')

plt.title('Training and Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot_a_p.png')
# plt.savefig('test_loss_plot.png')


