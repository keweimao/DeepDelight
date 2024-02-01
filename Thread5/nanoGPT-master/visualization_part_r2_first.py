import json
import matplotlib.pyplot as plt

# with open('train_losses_a2r0.json', 'r') as file:
#     train_losses_a2r0 = json.load(file)

with open('val_losses_a2r0_2.json', 'r') as file:
    val_losses_a2r0_2 = json.load(file)

# with open('train_losses_a2r1.json', 'r') as file:
#     train_losses_a2r1 = json.load(file)

with open('val_losses_a2r2_2.json', 'r') as file:
    val_losses_a2r2_2 = json.load(file)

# with open('train_losses_a2r2.json', 'r') as file:
#     train_losses_a2r2 = json.load(file)

with open('val_losses_a2r5_2.json', 'r') as file:
    val_losses_a2r5_2 = json.load(file)

# with open('train_losses_origin.json', 'r') as file:
#     train_losses_origin = json.load(file)

# with open('val_losses_origin.json', 'r') as file:
#     val_losses_origin = json.load(file)



# # # Plotting
# plt.figure(figsize=(10, 6))
# # plt.plot(train_losses_a2r0, label='Training Loss a2r0')
# plt.plot(val_losses_a2r0, label='Validation Loss a2r0')

# # plt.plot(train_losses_a2r1, label='Training Loss a2r1')
# plt.plot(val_losses_a2r1, label='Validation Loss a2r1')

# # plt.plot(train_losses_a2r2, label='Training Loss a2r2')
# plt.plot(val_losses_a2r2, label='Validation Loss a2r2')

# # plt.plot(train_losses_origin, label='Training Loss Origin')
# # plt.plot(val_losses_origin, label='Validation Loss Origin')

# plt.title('Training and Validation Loss over Iterations')
# plt.xlabel('Iteration Number')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.savefig('loss_plot.png')
# # plt.savefig('test_loss_plot.png')


# # Plotting
plt.figure(figsize=(10, 6))
# plt.plot(train_losses_a2r0, label='Training Loss a2r0')
plt.plot(val_losses_a2r0_2, label='Validation Loss a2r0')

# plt.plot(train_losses_a2r1, label='Training Loss a2r1')
plt.plot(val_losses_a2r2_2, label='Validation Loss a2r1')

# plt.plot(train_losses_a2r2, label='Training Loss a2r2')
plt.plot(val_losses_a2r5_2, label='Validation Loss a2r2')

# plt.plot(train_losses_origin, label='Training Loss Origin')
# plt.plot(val_losses_origin, label='Validation Loss Origin')




plt.title('Training and Validation Loss over Iterations')
plt.xlabel('Iteration Number')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
# plt.show()

# Focus on the latter half of the iterations
num_iterations = len(val_losses_a2r2_2)  # Use the actual number of iterations
start_point = num_iterations // 100  # Starting from the halfway point

# plt.xlim(start_point, num_iterations)  # Set x-axis limits to focus on the latter half
plt.xlim(0, start_point)  # Set x-axis limits to focus on the latter half

# Adjusting the y-axis range
# Calculate the maximum and minimum loss values in the focused range
# max_loss = max(max(val_losses_a2r0_2[start_point:]),
#                max(val_losses_a2r2_2[start_point:]),
#                max(val_losses_a2r5_2[start_point:]))


# min_loss = min(min(val_losses_a2r0_2[start_point:]),
#                min(val_losses_a2r2_2[start_point:]),
#                min(val_losses_a2r5_2[start_point:]))

max_loss = max(max(val_losses_a2r0_2[:start_point]),
               max(val_losses_a2r2_2[:start_point]),
               max(val_losses_a2r5_2[:start_point]))


min_loss = min(min(val_losses_a2r0_2[:start_point]),
               min(val_losses_a2r2_2[:start_point]),
               min(val_losses_a2r5_2[:start_point]))

plt.ylim(min_loss, max_loss)  # Set y-axis limits to enhance the loss variation

# Display the plot
plt.savefig('loss_plot_part_r2_first.png')