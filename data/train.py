import torch
import torch.optim as optim
from encoder import Autoencoder
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from classifier import Classifier


# Transform to normalize the data
transform = transforms.Compose([transforms.ToTensor()])
# transform = transform+1
# transform = transform/2
# Load datasets
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
# print(len(train_dataset))
# Data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# # Initialize the model, loss function and optimizer
# model = Autoencoder().to(device)
# criterion = nn.L1Loss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
#
# # Training loop
# num_epochs = 20
#
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0
#     for images, _ in train_loader:
#         images = images.to(device)
#         optimizer.zero_grad()
#         outputs = model(images)
#         loss = criterion(outputs, images)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * images.size(0)
#
#     train_loss /= len(train_loader.dataset)
#     # print(len(train_loader.dataset))
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}')
#
#     # Validation
#     model.eval()
#     val_loss = 0
#     with torch.no_grad():
#         for images, _ in test_loader:
#             images = images.to(device)
#             outputs = model(images)
#             loss = criterion(outputs, images)
#             val_loss += loss.item() * images.size(0)
#
#     val_loss /= len(test_loader.dataset)
#     print(f'Validation Loss: {val_loss:.4f}')
#
#     model.eval()
#     with torch.no_grad():
#         for images, _ in test_loader:
#             images = images.to(device)
#             outputs = model(images)
#             break  # We just want one batch for visualization
#
#     # Move images to CPU and denormalize for plotting
#     images = images.cpu().numpy()
#     outputs = outputs.cpu().numpy()
#
#     # Plot original images
#     fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
#     for i in range(10):
#         axes[i].imshow(images[i].reshape(28, 28), cmap='gray')
#         axes[i].axis('off')
#     fig.suptitle('Original Images')
#     plt.show()
#
#     # Plot reconstructed images
#     fig, axes = plt.subplots(1, 10, figsize=(15, 1.5))
#     for i in range(10):
#         axes[i].imshow(outputs[i].reshape(28, 28), cmap='gray')
#         axes[i].axis('off')
#     fig.suptitle('Reconstructed Images')
#     plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model, loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 20
# train_losses, val_losses = [], []
# train_accuracies, val_accuracies = []

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    correct_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_train += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = correct_train / len(train_loader.dataset)
    # train_losses.append(train_loss)
    # train_accuracies.append(train_accuracy)

    # Validation
    model.eval()
    val_loss = 0
    correct_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_val += (predicted == labels).sum().item()

    val_loss /= len(test_loader.dataset)
    val_accuracy = correct_val / len(test_loader.dataset)
    # val_losses.append(val_loss)
    # val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
