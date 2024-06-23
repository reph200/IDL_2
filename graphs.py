import matplotlib.pyplot as plt

# Data from the provided results
# epochs = list(range(1, 21))
# train_losses = [0.1340, 0.0589, 0.0434, 0.0402, 0.0385,
#                 0.0373, 0.0366, 0.0360, 0.0355, 0.0350,
#                 0.0347, 0.0344, 0.0341, 0.0338, 0.0336,
#                 0.0334, 0.0333, 0.0331, 0.0330, 0.0329]
# valid_losses = [0.1086, 0.0453, 0.0412, 0.0385, 0.0374,
#                 0.0363, 0.0357, 0.0353, 0.0353, 0.0346,
#                 0.0345, 0.0339, 0.0339, 0.0336, 0.0334,
#                 0.0333, 0.0332, 0.0330, 0.0330, 0.0327]
#
# # Plotting the training and validation losses
# plt.figure(figsize=(10, 6))
# # plt.plot(epochs, train_losses, marker='o', label='Training Loss')
# plt.plot(epochs, valid_losses, marker='o', label='Validation Loss')
# plt.title('Validation Losses')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.xticks(epochs)
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# import matplotlib.pyplot as plt

# Data provided
epochs = list(range(1, 21))
train_losses = [0.3332, 0.0941, 0.0693, 0.0552, 0.0455, 0.0394, 0.0333, 0.0292, 0.0235, 0.0225,
                0.0185, 0.0175, 0.0140, 0.0141, 0.0140, 0.0107, 0.0109, 0.0116, 0.0090, 0.0086]
val_losses = [0.1156, 0.0658, 0.0546, 0.0516, 0.0517, 0.0499, 0.0469, 0.0428, 0.0599, 0.0511,
              0.0680, 0.0589, 0.0571, 0.0695, 0.0634, 0.0593, 0.0692, 0.0630, 0.0602, 0.0734]
train_accuracies = [0.8965, 0.9715, 0.9788, 0.9824, 0.9855, 0.9868, 0.9895, 0.9902, 0.9921, 0.9921,
                    0.9939, 0.9943, 0.9952, 0.9951, 0.9949, 0.9964, 0.9960, 0.9963, 0.9968, 0.9972]
val_accuracies = [0.9642, 0.9798, 0.9827, 0.9838, 0.9830, 0.9832, 0.9847, 0.9864, 0.9822, 0.9849,
                  0.9817, 0.9841, 0.9846, 0.9818, 0.9829, 0.9839, 0.9847, 0.9859, 0.9850, 0.9838]

# Plotting training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, label='Train Accuracy')
plt.plot(epochs, val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
