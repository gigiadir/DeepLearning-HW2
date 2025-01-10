import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lstm_autoencoder import LSTMAutoEncoder

def train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist", should_save_model = True):
    transform = transforms.Compose([transforms.ToTensor()])
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = LSTMAutoEncoder(args.input_size, args.hidden_size, args.n_classes)
    model.train()

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    is_classification = bool(args.n_classes > 0)
    epochs = args.epochs

    test_accuracy_per_epoch = []
    loss_per_epoch = []

    model.eval()
    correct_predictions = 0
    total_samples = 0
    train_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(images.shape[0], args.input_size, args.input_size)
            reconstruction, classes_probabilities = model(images)
            predictions = torch.argmax(classes_probabilities, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

        test_accuracy_per_epoch.append(correct_predictions / total_samples)

        for images, labels in train_loader:
            images = images.view(images.shape[0], args.input_size, args.input_size)
            reconstruction, classes_probabilities = model(images)

            if is_classification:
                loss = classification_criterion(classes_probabilities, labels)
            else:
                loss = reconstruction_criterion(reconstruction, images)

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_per_epoch.append(avg_train_loss)


    for epoch in range(epochs):
        model.train()
        train_loss = 0

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

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        loss_per_epoch.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss}")

        model.eval()
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.view(images.shape[0], args.input_size, args.input_size)
                reconstruction, classes_probabilities = model(images)
                predictions = torch.argmax(classes_probabilities, dim=1)
                correct_predictions += (predictions == labels).sum().item()
                total_samples += labels.size(0)

        test_accuracy_per_epoch.append(correct_predictions / total_samples)

    if should_save_model:
        torch.save(model.state_dict(), f"output/mnist/models/{model_name}.pth")

    return model, loss_per_epoch, test_accuracy_per_epoch

def generate_accuracy_and_loss_plot(epochs, loss_per_epoch, test_accuracy_per_epoch):
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))

    axs[0].plot(epochs, loss_per_epoch, label="Loss", marker='o')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Loss vs. Epochs')
    axs[0].set_xticks(epochs)
    axs[0].set_ylim(0, None)
    axs[0].grid()
    axs[0].legend()

    axs[1].plot(epochs, test_accuracy_per_epoch, label='Accuracy', marker='s', color='orange')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_title('Accuracy vs. Epochs')
    axs[1].set_xticks(epochs)
    axs[1].set_ylim(0, 1)
    axs[1].grid()
    axs[1].legend()

    last_epoch = epochs[-1]
    last_accuracy = test_accuracy_per_epoch[-1]
    axs[1].scatter(last_epoch, last_accuracy, color='darkorange', s=200, alpha=0.6, marker='s',
                   zorder=3)
    axs[1].annotate(f"{last_accuracy:.2f}", (last_epoch, last_accuracy), textcoords="offset points",
                    xytext=(-10, -15), ha='center')

    plt.tight_layout()
    plt.savefig("output/mnist/losses_vs_accuracy.png")
    plt.show()

def section_1(args):
    transform = transforms.Compose([transforms.ToTensor()])
    model = train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist", should_save_model=False)
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

    model, loss_per_epoch, test_accuracy_per_epoch = train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist_classification", should_save_model=True)
    epochs = list(range(args.epochs + 1))
    generate_accuracy_and_loss_plot(epochs, loss_per_epoch, test_accuracy_per_epoch)

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


