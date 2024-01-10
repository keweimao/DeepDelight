import json
import matplotlib.pyplot as plt

with open('train_losses.json', 'r') as file:
    train_losses = json.load(file)

with open('val_losses.json', 'r') as file:
    val_losses = json.load(file)

with open('train_losses_dropout.json', 'r') as file:
    train_losses_dropout = json.load(file)

with open('val_losses_dropout.json', 'r') as file:
    val_losses_dropout = json.load(file)

with open('train_losses_origin.json', 'r') as file:
    train_losses_origin = json.load(file)

with open('val_losses_origin.json', 'r') as file:
    val_losses_origin = json.load(file)



# # Plotting
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')

plt.plot(train_losses_dropout, label='Training Loss Dropout')
plt.plot(val_losses_dropout, label='Validation Loss Dropout')

# plt.plot(train_losses_origin, label='Training Loss Origin')
# plt.plot(val_losses_origin, label='Validation Loss Origin')

plt.title('Training and Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.savefig('loss_plot.png')
# plt.savefig('test_loss_plot.png')


