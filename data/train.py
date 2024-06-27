import torch
import torch.optim as optim
from autoencoder import Autoencoder
from autoencoder import Encoder
from autoencoder import Decoder
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from classifier import Classifier, FineTunedClassifier
import torch.utils.data as data_utils

if __name__ == '__main__':

    # Transform to normalize the data
    transform = transforms.Compose([transforms.ToTensor()])
    # transform = transform+1
    # transform = transform/2
    # Load datasets
    train_dataset = datasets.MNIST(root='./', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./', train=False, transform=transform, download=True)
    # print(len(train_dataset))
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    ########################################## Section a #####################

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    # # Initialize the model, loss function and optimizer
    # model = Autoencoder().to(device)
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    #
    # # Training loop
    # num_epochs = 20
    # train_losses = []
    # val_losses = []
    # all_reconstructed_images = []
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
    #     train_losses.append(train_loss)
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
    #     val_losses.append(val_loss)
    #     print(f'Validation Loss: {val_loss:.4f}')
    #
    #     # Collect reconstructed images
    #     with torch.no_grad():
    #         for images, _ in test_loader:
    #             images = images.to(device)
    #             outputs = model(images)
    #             all_reconstructed_images.append(outputs.cpu().numpy())
    #             break  # We just want one batch for visualization
    #
    # # Plotting the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    #
    # # Plotting all reconstructed images
    # num_epochs_shown = len(all_reconstructed_images)
    # fig, axes = plt.subplots(num_epochs_shown, 10, figsize=(15, 1.5 * num_epochs_shown))
    # for epoch_idx in range(num_epochs_shown):
    #     images = all_reconstructed_images[epoch_idx]
    #     for img_idx in range(10):
    #         axes[epoch_idx, img_idx].imshow(images[img_idx].reshape(28, 28), cmap='gray')
    #         axes[epoch_idx, img_idx].axis('off')
    #
    # fig.suptitle('Reconstructed Images Over Epochs')
    # plt.show()

    # # Save the entire Autoencoder model
    # torch.save(model.state_dict(), 'autoencoder_weights_1.pth')
    # print("Autoencoder weights saved to autoencoder_weights_1.pth")
    # # Save the Encoder separately
    # encoder_weights_path = './encoder_weights_section_1.pth'
    # torch.save(model.encoder.state_dict(), encoder_weights_path)
    # print(f"Encoder weights saved to {encoder_weights_path}")

    ############################################## Section b ######################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model, loss function, and optimizer
    model = Classifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

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
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)

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
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')

    # encoder_weights_path = './encoder_weights_2.pth'
    # torch.save(model.encoder.state_dict(), encoder_weights_path)
    # print(f"Encoder weights saved to {encoder_weights_path}")

    # Plotting the training and validation loss
    plt.figure(figsize=(15, 5))

    # Plot the loss
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.grid(True)

    # Plot the accuracy
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    ######################################### Section c ############################################

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    #
    # # Initialize the encoder, load the saved weights
    # encoder = Encoder().to(device)
    # encoder.load_state_dict(torch.load('./encoder_weights_2.pth'))
    #
    # # Initialize the decoder
    # decoder = Decoder().to(device)
    #
    # # Define the loss function and optimizer
    # criterion = nn.L1Loss()
    # optimizer = optim.Adam(decoder.parameters(), lr=0.001)
    #
    # num_epochs = 20
    # train_losses = []
    # val_losses = []
    #
    # for epoch in range(num_epochs):
    #     decoder.train()
    #     train_loss = 0
    #     for images, _ in train_loader:
    #         images = images.to(device)
    #         optimizer.zero_grad()
    #         encoded_images = encoder(images)
    #         outputs = decoder(encoded_images)
    #         loss = criterion(outputs, images)
    #         loss.backward()
    #         optimizer.step()
    #         train_loss += loss.item() * images.size(0)
    #
    #     train_loss /= len(train_loader.dataset)
    #     train_losses.append(train_loss)
    #     print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}')
    #
    #     # Validation
    #     decoder.eval()
    #     val_loss = 0
    #     with torch.no_grad():
    #         for images, _ in test_loader:
    #             images = images.to(device)
    #             encoded_images = encoder(images)
    #             outputs = decoder(encoded_images)
    #             loss = criterion(outputs, images)
    #             val_loss += loss.item()* images.size(0)
    #
    #     val_loss /= len(test_loader.dataset)
    #     val_losses.append(val_loss)
    #     print(f'Validation Loss: {val_loss:.4f}')
    #
    #     # Visualize some test images and their reconstructions
    #     with torch.no_grad():
    #         for images, _ in test_loader:
    #             images = images.to(device)
    #             encoded_images = encoder(images)
    #             outputs = decoder(encoded_images)
    #             break
    #
    #     images = images.cpu().numpy()
    #     outputs = outputs.cpu().numpy()
    #
    # torch.save(decoder.state_dict(), './decoder_weights_3.pth')
    # print(f"Decoder weights saved to ./decoder_weights_3.pth")
    #
    # # Plotting the training and validation loss
    # plt.figure(figsize=(10, 5))
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    # plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss over Epochs')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    ######################################### Section d ############################################

    # import torch
    # import torch.optim as optim
    # import torch.nn as nn
    # from torchvision import datasets, transforms
    # from torch.utils.data import DataLoader, Subset
    # import torch.utils.data as data_utils
    # import matplotlib.pyplot as plt
    # from classifier import Classifier  # Assuming Classifier is implemented in classifier.py
    #
    # if __name__ == '__main__':
    #     # Device configuration
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {device}")
    #
    #     # Define transformations
    #     transform = transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5,), (0.5,))
    #     ])
    #
    #     # Load MNIST dataset
    #     train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    #     test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    #
    #     # Create a subset of 100 labeled examples
    #     indices = torch.randperm(len(train_dataset))[:100]  # Randomly select 100 indices
    #     train_loader_CLS = DataLoader(data_utils.Subset(train_dataset, indices), batch_size=64, shuffle=True)
    #     test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    #     # Initialize classifier model
    #     classifier = Classifier()
    #
    #     # Define loss function and optimizer
    #     criterion = nn.CrossEntropyLoss()
    #     optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    #
    #     # Move model to appropriate device (GPU if available)
    #     classifier.to(device)
    #
    #     # Training and testing loop
    #     num_epochs = 20
    #     train_losses = []
    #     train_accuracies = []
    #     test_losses = []
    #     test_accuracies = []
    #     test_epochs = []
    #
    #     for epoch in range(num_epochs):
    #         # Training phase
    #         classifier.train()
    #         running_train_loss = 0.0
    #         correct_train = 0
    #         total_train = 0
    #
    #         for images, labels in train_loader_CLS:
    #             images, labels = images.to(device), labels.to(device)
    #             optimizer.zero_grad()
    #             outputs = classifier(images)
    #             loss = criterion(outputs, labels)
    #             loss.backward()
    #             optimizer.step()
    #
    #             running_train_loss += loss.item()  # Accumulate the batch loss
    #
    #             _, predicted_train = torch.max(outputs, 1)
    #             total_train += labels.size(0)
    #             correct_train += (predicted_train == labels).sum().item()
    #
    #         # Calculate average training loss and accuracy for the epoch
    #         train_loss = running_train_loss / len(train_loader_CLS.dataset)
    #         train_accuracy = correct_train / len(train_loader_CLS.dataset)
    #
    #         train_losses.append(train_loss)
    #         train_accuracies.append(train_accuracy)
    #
    #         print(
    #             f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
    #
    #         # Validation phase
    #         if epoch % 1 == 0:
    #             classifier.eval()
    #             running_test_loss = 0.0
    #             correct_test = 0
    #             total_test = 0
    #
    #             with torch.no_grad():
    #                 for images, labels in test_loader:
    #                     images, labels = images.to(device), labels.to(device)
    #                     outputs = classifier(images)
    #                     loss = criterion(outputs, labels)
    #
    #                     running_test_loss += loss.item()  # Accumulate the batch loss
    #
    #                     _, predicted_test = torch.max(outputs, 1)
    #                     total_test += labels.size(0)
    #                     correct_test += (predicted_test == labels).sum().item()
    #
    #             # Calculate average test loss and accuracy for the epoch
    #             test_loss = running_test_loss / len(test_loader.dataset)
    #             test_accuracy = correct_test / len(test_loader.dataset)
    #
    #             test_losses.append(test_loss)
    #             test_accuracies.append(test_accuracy)
    #             test_epochs.append(epoch + 1)  # Store the epoch number for test evaluations
    #
    #             print(f'Test - Epoch [{epoch + 1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
    #
    #     print('Finished Training and Testing')
    #
    #     # Plotting the training and test progress
    #     plt.figure(figsize=(15, 5))
    #
    #     # Plot the loss
    #     plt.subplot(1, 2, 1)
    #     plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    #     plt.plot(test_epochs, test_losses, label='Test Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Training and Test Loss over Epochs')
    #     plt.legend()
    #
    #     # Plot the accuracy
    #     plt.subplot(1, 2, 2)
    #     plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    #     plt.plot(test_epochs, test_accuracies, label='Test Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.title('Training and Test Accuracy over Epochs')
    #     plt.legend()
    #
    #     plt.tight_layout()
    #     plt.show()
    #
    #     # Save the encoder weights (assuming classifier has an encoder attribute)
    #     encoder_weights_path = './encoder_weights_4.pth'
    #     torch.save(classifier.encoder.state_dict(), encoder_weights_path)
    #     print(f"Encoder weights saved to {encoder_weights_path}")

    ######################################## Section e ############################################

    # # Device configuration
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")
    #
    # # Define transformations
    # transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.5,), (0.5,))
    # ])
    #
    # # Load MNIST dataset
    # train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    # test_dataset = datasets.MNIST("data", train=False, download=True, transform=transform)
    #
    # # Create a subset of 100 labeled examples
    # indices = torch.randperm(len(train_dataset))[:100]  # Randomly select 100 indices
    # train_loader_CLS = DataLoader(data_utils.Subset(train_dataset, indices), batch_size=64, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    #
    # # Initialize classifier model
    # classifier = FineTunedClassifier()
    #
    # # Load pre-trained encoder weights
    # encoder_weights_path = './encoder_weights_section_1.pth'
    # encoder_dict = torch.load(encoder_weights_path, map_location=device)
    #
    # # Update the encoder weights in the classifier
    # classifier.encoder.load_state_dict(encoder_dict)
    #
    # # Define loss function and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(classifier.fc.parameters(), lr=0.001)
    #
    # # Move model to appropriate device (GPU if available)
    # classifier.to(device)
    #
    # # Training and testing loop
    # num_epochs = 20
    # train_losses = []
    # train_accuracies = []
    # test_losses = []
    # test_accuracies = []
    # test_epochs = []
    #
    # for epoch in range(num_epochs):
    #     # Training phase
    #     classifier.train()
    #     running_train_loss = 0.0
    #     correct_train = 0
    #     total_train = 0
    #
    #     for images, labels in train_loader_CLS:
    #         images, labels = images.to(device), labels.to(device)
    #
    #         optimizer.zero_grad()
    #         outputs = classifier(images)
    #         loss = criterion(outputs, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         running_train_loss += loss.item()  # Accumulate the batch loss
    #
    #         _, predicted_train = torch.max(outputs, 1)
    #         total_train += labels.size(0)
    #         correct_train += (predicted_train == labels).sum().item()
    #
    #     # Calculate average training loss and accuracy for the epoch
    #     train_loss = running_train_loss / len(train_loader_CLS.dataset)
    #     train_accuracy = correct_train / len(train_loader_CLS.dataset)
    #
    #     train_losses.append(train_loss)
    #     train_accuracies.append(train_accuracy)
    #
    #     print(
    #         f'Training - Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}')
    #
    #     # Validation phase
    #     if epoch % 1 == 0:
    #         classifier.eval()
    #         running_test_loss = 0.0
    #         correct_test = 0
    #         total_test = 0
    #
    #         with torch.no_grad():
    #             for images, labels in test_loader:
    #                 images, labels = images.to(device), labels.to(device)
    #                 outputs = classifier(images)
    #                 loss = criterion(outputs, labels)
    #
    #                 running_test_loss += loss.item()  # Accumulate the batch loss
    #
    #                 _, predicted_test = torch.max(outputs, 1)
    #                 total_test += labels.size(0)
    #                 correct_test += (predicted_test == labels).sum().item()
    #
    #         # Calculate average test loss and accuracy for the epoch
    #         test_loss = running_test_loss / len(test_loader.dataset)
    #         test_accuracy = correct_test / len(test_loader.dataset)
    #
    #         test_losses.append(test_loss)
    #         test_accuracies.append(test_accuracy)
    #         test_epochs.append(epoch + 1)  # Store the epoch number for test evaluations
    #
    #         print(f'Test - Epoch [{epoch + 1}/{num_epochs}], Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}')
    #
    # print('Finished Training and Testing')
    #
    # # Plotting the training and test progress
    # plt.figure(figsize=(15, 5))
    #
    # # Plot the loss
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    # plt.plot(test_epochs, test_losses, label='Test Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Test Loss over Epochs')
    # plt.legend()
    #
    # # Plot the accuracy
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
    # plt.plot(test_epochs, test_accuracies, label='Test Accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.title('Training and Test Accuracy over Epochs')
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
