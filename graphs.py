import matplotlib.pyplot as plt

# Data from the provided results
epochs = list(range(1, 21))
train_losses = [0.1340, 0.0589, 0.0434, 0.0402, 0.0385,
                0.0373, 0.0366, 0.0360, 0.0355, 0.0350,
                0.0347, 0.0344, 0.0341, 0.0338, 0.0336,
                0.0334, 0.0333, 0.0331, 0.0330, 0.0329]
valid_losses = [0.1086, 0.0453, 0.0412, 0.0385, 0.0374,
                0.0363, 0.0357, 0.0353, 0.0353, 0.0346,
                0.0345, 0.0339, 0.0339, 0.0336, 0.0334,
                0.0333, 0.0332, 0.0330, 0.0330, 0.0327]

# Plotting the training and validation losses
plt.figure(figsize=(10, 6))
# plt.plot(epochs, train_losses, marker='o', label='Training Loss')
plt.plot(epochs, valid_losses, marker='o', label='Validation Loss')
plt.title('Validation Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.xticks(epochs)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()