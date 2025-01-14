import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from lstm_autoencoder import LSTMAutoEncoder
from utils import get_optimizer


def prepare_images(images, is_pixel_series, n_row = 28, n_col = 28):
    if is_pixel_series:
        images = images.view(images.shape[0], -1, 1)
    else:
        images = images.view(images.shape[0], n_row, n_col)

    return images

def get_transform(args, is_pixel_series):
    if is_pixel_series:
        transform = transforms.Compose([transforms.Resize((args.input_size ** 2, 1)), transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor()])

    return transform

def calculate_classification_accuracy(model, data_loader, is_pixel_series, args):
    model.eval()
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = prepare_images(images, is_pixel_series, args.input_size, args.input_size)
            reconstruction, classes_probabilities = model(images)
            predictions = torch.argmax(classes_probabilities, dim=1)
            correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    accuracy = correct / total_samples

    return accuracy

def calculate_loss_on_dataset(model, data_loader, is_pixel_series, args):
    model.eval()
    classification_criterion = nn.CrossEntropyLoss()
    reconstruction_criterion = nn.MSELoss()

    is_classification = bool(model.n_classes > 0)

    total_loss = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images = prepare_images(images, is_pixel_series, args.input_size, args.input_size)
            reconstruction, classes_probabilities = model(images)

            if is_classification:
                loss = classification_criterion(classes_probabilities, labels)
            else:
                loss = reconstruction_criterion(reconstruction, images)
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
    return avg_loss


def train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist",
                                     should_save_model = True, is_pixel_series = False):

    transform = get_transform(args, is_pixel_series)
    train_set = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

    model = LSTMAutoEncoder(args.input_size, args.hidden_size, args.n_classes)
    model.train()

    reconstruction_criterion = nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, args)

    is_classification = bool(args.n_classes > 0)
    epochs = args.epochs

    test_accuracy_per_epoch = []
    loss_per_epoch = []

    if is_classification:
        initial_accuracy = calculate_classification_accuracy(model, test_loader, is_pixel_series, args)
        test_accuracy_per_epoch.append(initial_accuracy)
        print(f"Initial accuracy: {initial_accuracy}")


    initial_loss = calculate_loss_on_dataset(model, train_loader, is_pixel_series, args)
    loss_per_epoch.append(initial_loss)
    print(f"Initial loss: {initial_loss}")

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        for images, labels in train_loader:
            model.train()
            optimizer.zero_grad()
            images = prepare_images(images, is_pixel_series, args.input_size, args.input_size)
            reconstruction, classes_probabilities = model(images)

            if is_classification:
                loss = classification_criterion(classes_probabilities, labels)
            else:
                loss = reconstruction_criterion(reconstruction, images)

            loss.backward()

            # for name, param in model.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"Gradient for {name}: {param.grad.norm().item():.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clipping)

            # total_norm = 0.0
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         param_norm = param.grad.data.norm(2)
            #         total_norm += param_norm.item() ** 2
            # total_norm = total_norm ** 0.5
            # print(f"Epoch [{epoch + 1}/{epochs}], Batch {batch_count}, Gradient Norm: {total_norm:.4f}")

            optimizer.step()

            train_loss += loss.item()
            batch_count += 1
            print(f"Epoch [{epoch + 1}/{epochs}], Batch {batch_count}, Total Loss: {train_loss}")

        avg_train_loss = train_loss / len(train_loader)
        loss_per_epoch.append(avg_train_loss)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_train_loss}")

        test_accuracy = calculate_classification_accuracy(model, test_loader, is_pixel_series, args)
        test_accuracy_per_epoch.append(test_accuracy)
        print(f"Epoch [{epoch + 1}/{epochs}], Test Accuracy: {test_accuracy}")


    if should_save_model:
        torch.save(model.state_dict(), f"output/mnist/models/{model_name}.pth")

    return model, loss_per_epoch, test_accuracy_per_epoch

def generate_accuracy_and_loss_plot(epochs, loss_per_epoch, test_accuracy_per_epoch, filename="losses_vs_accuracy"):
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
    plt.savefig(f"output/mnist/{filename}.png")
    plt.show()

def section_1(args):
    transform = transforms.Compose([transforms.ToTensor()])
    model, _, _ = train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist", should_save_model=False)
    # model = LSTMAutoEncoder(args.input_size, args.hidden_size)
    # model.load_state_dict(torch.load("output/mnist/models/lstm_autoencoder_mnist.pth", weights_only=True))

    test_set = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)


    with torch.no_grad():
        images, _ = next(iter(test_loader))
        images = images.view(images.size(0), args.input_size, args.input_size)
        reconstructed, _= model(images)
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

    model, loss_per_epoch, test_accuracy_per_epoch = train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist_classification", should_save_model=False)
    epochs = list(range(args.epochs + 1))
    generate_accuracy_and_loss_plot(epochs, loss_per_epoch, test_accuracy_per_epoch)

def section_3(args):
    args.n_classes = 10
    args.epochs = 10
    args.input_size = 1
    args.gradient_clipping = 1.0
    args.batch_size = 128
    args.learning_rate = 1e-3
    args.hidden_size = 64

    model, loss_per_epoch, test_accuracy_per_epoch = train_and_evaluate_lstm_ae_mnist(args, model_name="lstm_autoencoder_mnist_classification_pixel_by_pixel", should_save_model=True, is_pixel_series=True)
    epochs = list(range(args.epochs + 1))
    generate_accuracy_and_loss_plot(epochs, loss_per_epoch, test_accuracy_per_epoch, filename="loss_vs_accuracy_pixel_series")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=28)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--n-classes", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")

    args = parser.parse_args()

    section_1(args)


