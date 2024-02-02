import json
import matplotlib.pyplot as plt


with open('val_losses_a2r0_2_p.json', 'r') as file:
    val_losses_a2r0_2 = json.load(file)


with open('val_losses_a2r2_2_p.json', 'r') as file:
    val_losses_a2r2_2 = json.load(file)


with open('val_losses_a2r5_2_p.json', 'r') as file:
    val_losses_a2r5_2 = json.load(file)

# with open('train_losses_a2r10.json', 'r') as file:
#     train_losses_a2r10 = json.load(file)

# with open('val_losses_a2r10.json', 'r') as file:
#     val_losses_a2r10 = json.load(file)






# # # Plotting
# plt.figure(figsize=(10, 6))
# # plt.plot(train_losses_a2r0, label='Training Loss a2r0')
    
plt.plot(val_losses_a2r0_2, label='Validation Loss a2r0_2')

plt.plot(val_losses_a2r2_2, label='Validation Loss a2r2_2')

plt.plot(val_losses_a2r5_2, label='Validation Loss a2r5_2')

# plt.plot(val_losses_a2r10, label='Validation Loss a2r10')

plt.title('Training and Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot_r_2_p.png')
# plt.savefig('test_loss_plot.png')


