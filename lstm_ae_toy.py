import argparse

import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def generate_synthetic_data(n_seq, seq_len):
    data = np.random.rand(n_seq, seq_len)

    for seq in data:
        i = np.random.randint(20, 31)
        seq[i-5:i+5] *= 0.1

    return data

def split_data_to_train_validation_test(data, train_percentage = 0.6, validation_percentage = 0.2):
    train_size = int(len(data) * train_percentage)
    validation_size = int(len(data) * validation_percentage)
    train_data, validation_data, test_data = np.split(data, [train_size, train_size + validation_size])

    return train_data, validation_data, test_data

def plot_examples(data: np.ndarray):
    total_num_samples = data.shape[0]
    num_samples_to_display = 3

    indices = np.random.choice(total_num_samples, num_samples_to_display, replace=False)

    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.ylim(0, 1.25)
    for i in indices:
        plt.plot(data[i], label=f"Example {i}")
    plt.legend(loc="upper right")
    plt.title(f"Synthetic Data - {num_samples_to_display} Example Signals")
    plt.savefig("./output/synthetic_data/section_1/example.png")
    plt.show()


class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_AE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.encoder = nn.LSTM(input_size = input_size, hidden_size = hidden_size, num_layers=1, batch_first=True)
        self.decoder = nn.LSTM(input_size = hidden_size, hidden_size = hidden_size, num_layers=1, batch_first=True, proj_size=1)

    def forward(self, x):
        _, (context, _) = self.encoder(x)
        context = context.repeat(x.size(1), 1, 1).permute(1, 0, 2)
        output, _ = self.decoder(context)

        return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type = int, default = 32)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1000)

    args = parser.parse_args()

    n_seq = 10000
    seq_len = 50
    data = generate_synthetic_data(n_seq = n_seq, seq_len = seq_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    #     Section 1
    #plot_examples(data)

    #     Section 2

    training_data, validation_data, test_data = split_data_to_train_validation_test(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_tensor = torch.from_numpy(training_data).float().unsqueeze(-1).to(device)
    validation_tensor = torch.from_numpy(validation_data).float().unsqueeze(-1).to(device)
    test_tensor = torch.from_numpy(test_data).float().unsqueeze(-1).to(device)

    learning_rate = args.learning_rate
    epochs = args.epochs
    hidden_size = args.hidden_size
    batch_size = args.batch_size

    training_loader = DataLoader(TensorDataset(training_tensor), batch_size = batch_size, shuffle = True)

    model = LSTM_AE(input_size = 1, hidden_size = hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (inputs, ) in enumerate(training_loader):
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(training_loader):.4f}")


    validation_loader = DataLoader(TensorDataset(validation_tensor), batch_size = 2, shuffle = False)

    model.eval()
    with torch.no_grad():
        inputs, = next(iter(validation_loader))
        outputs = model(inputs)

    original = inputs[0].cpu().numpy()
    reconstructed = outputs[0].cpu().numpy()


    timesteps = range(original.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(timesteps, original[:, 0], label="Original (Feature 0)", alpha=0.8)
    plt.plot(timesteps, reconstructed[:, 0], label="Reconstructed (Feature 0)", alpha=0.8, linestyle="--")
    plt.title("Original vs. Reconstructed Sequence (Feature 0)")
    plt.xlabel("Timesteps")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.show()



