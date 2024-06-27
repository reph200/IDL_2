import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from autoencoder import Encoder, Decoder, Autoencoder


# Function to visualize images
def visualize_images(original, autoencoder_reconstructed, encoder_decoder_reconstructed, num_images=10):
    fig, axes = plt.subplots(3, num_images, figsize=(15, 5))
    for i in range(num_images):
        axes[0, i].imshow(original[i].reshape(28, 28), cmap='gray')
        axes[0, i].axis('off')
        axes[1, i].imshow(autoencoder_reconstructed[i].reshape(28, 28), cmap='gray')
        axes[1, i].axis('off')
        axes[2, i].imshow(encoder_decoder_reconstructed[i].reshape(28, 28), cmap='gray')
        axes[2, i].axis('off')
    axes[0, 0].set_title('Original Images')
    axes[1, 0].set_title('Autoencoder Reconstructed')
    axes[2, 0].set_title('Classifier-Decoder Reconstructed')
    plt.show()


def collect_digit_examples(data_loader, num_examples=10):
    digit_examples = {i: [] for i in range(10)}
    for images, labels in data_loader:
        for img, lbl in zip(images, labels):
            if len(digit_examples[lbl.item()]) < num_examples:
                digit_examples[lbl.item()].append(img)
        if all(len(digit_examples[d]) >= num_examples for d in range(10)):
            break

    # Convert list of images to tensor
    for d in range(10):
        digit_examples[d] = torch.stack(digit_examples[d])

    return digit_examples


if __name__ == '__main__':
    # Define the transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # Load the MNIST dataset
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    autoencoder = Autoencoder().to(device)
    autoencoder.load_state_dict(torch.load('./data/autoencoder_weights_1.pth'))
    autoencoder.eval()

    encoder = Encoder().to(device)
    encoder.load_state_dict(torch.load('./data/encoder_weights_2.pth'))
    encoder.eval()

    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load('./data/decoder_weights_3.pth'))
    decoder.eval()

    # Part 1: Compare 50 different images
    collected_images = []
    for images, _ in test_loader:
        if len(collected_images) >= 50:
            break
        collected_images.append(images)

    images = torch.cat(collected_images, dim=0)[:50].to(device)

    # Autoencoder reconstruction
    with torch.no_grad():
        autoencoder_outputs = autoencoder(images)

    # Encoder-decoder reconstruction
    with torch.no_grad():
        encoded_images = encoder(images)
        encoder_decoder_outputs = decoder(encoded_images)

    # Move images to CPU and convert to numpy
    images = images.cpu().numpy()
    autoencoder_outputs = autoencoder_outputs.cpu().numpy()
    encoder_decoder_outputs = encoder_decoder_outputs.cpu().numpy()

    # Visualize original, autoencoder-reconstructed, and encoder-decoder reconstructed images
    visualize_images(images, autoencoder_outputs, encoder_decoder_outputs, num_images=10)

    # Part 2: Visualize 10 examples of each digit
    digit_examples = collect_digit_examples(test_loader, num_examples=10)

    for digit in range(10):
        images = digit_examples[digit].to(device)

        # Autoencoder reconstruction
        with torch.no_grad():
            autoencoder_outputs = autoencoder(images)

        # Encoder-decoder reconstruction
        with torch.no_grad():
            encoded_images = encoder(images)
            encoder_decoder_outputs = decoder(encoded_images)

        # Move images to CPU and convert to numpy
        images = images.cpu().numpy()
        autoencoder_outputs = autoencoder_outputs.cpu().numpy()
        encoder_decoder_outputs = encoder_decoder_outputs.cpu().numpy()

        # Visualize original, autoencoder-reconstructed, and encoder-decoder reconstructed images
        print(f"Digit: {digit}")
        visualize_images(images, autoencoder_outputs, encoder_decoder_outputs, num_images=10)
