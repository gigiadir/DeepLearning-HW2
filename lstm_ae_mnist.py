import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lstm_autoencoder import LSTMAutoEncoder

def train_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist", should_save_model = True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = LSTMAutoEncoder(args.input_size, args.hidden_size, args.n_classes)
    model.train()

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    is_classification = bool(args.n_classes > 0)
    epochs = args.epochs

    loss = 0.0
    total = 0
    correct = 0

    for epoch in range(epochs):
        model.train()

        for images, labels in train_loader:
            images = images.view(images.shape[0], args.input_size, args.input_size)

            optimizer.zero_grad()

            reconstruction, classes_probabilities = model(images)

            if is_classification:
                loss = classification_criterion(classes_probabilities, labels)
            else:
                loss = reconstruction_criterion(reconstruction, images)

            loss.backward()
            optimizer.step()

            loss += loss.item()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss / len(train_loader)}")

    if should_save_model:
        torch.save(model.state_dict(), f"output/mnist/models/{model_name}.pth")

    return model

def section_1(args):
    transform = transforms.Compose([transforms.ToTensor()])
    model = train_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist", should_save_model=False)
    # model = LSTMAutoEncoder(args.input_size, args.hidden_size)
    # model.load_state_dict(torch.load("output/mnist/models/lstm_autoencoder_mnist.pth", weights_only=True))

    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.view(images.size(0), args.input_size, args.input_size)
        reconstructed, _ = model(images)
        fig, axes = plt.subplots(2, 3, figsize=(10, 6))
        plt.suptitle('MNIST Original vs. Reconstructed', fontsize = 16)
        for i in range(3):
            axes[0, i].imshow(images[i].numpy().squeeze(), cmap='gray')
            axes[0, i].axis('off')
            axes[0, i].set_title("Original", fontsize = 14)
            axes[1, i].imshow(reconstructed[i].numpy().squeeze(), cmap='gray')
            axes[1, i].axis('off')
            axes[1, i].set_title("Reconstructed", fontsize = 14)
        #plt.savefig("output/mnist/mnist_original_vs_reconstruced.png")
        plt.show()

def section_2(args):
    args.n_classes = 10
    args.epochs = 10
    transform = transforms.Compose([transforms.ToTensor()])
    model = LSTMAutoEncoder(args.input_size, args.hidden_size, args.n_classes)
    model.load_state_dict(torch.load("output/mnist/models/lstm_autoencoder_mnist_classification.pth", weights_only=True))
    # model = train_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist_classification", should_save_model=True)
    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.view(images.shape[0], args.input_size, args.input_size)
        output, classes_probabilities = model(images)
        prediction = torch.argmax(classes_probabilities, dim=1)
        correct += (prediction == labels).sum().item()
        total += labels.size(0)

    print(f"Accuracy: {100 * correct / total}")
    # model = LSTMAutoEncoder(args.input_size, args.hidden_size, args.n_classes)
    #

    # train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    #
    # for images, labels in train_loader:
    #     images = images.view(images.shape[0], args.input_size, args.input_size)
    #     output, y_pred = model(images)
    #     prediction = torch.argmax(y_pred, dim=1)
    #     correct =  (prediction == labels).sum().item()
    #     print(correct)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-clipping", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=0)


    args = parser.parse_args()

    section_2(args)


