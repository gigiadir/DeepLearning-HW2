import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Step 1: Define the LSTM Autoencoder Model
class LSTMAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, input_size, batch_first=True)
        self.hidden_size = hidden_size

    def forward(self, x):
        # Encode
        _, (h_n, _) = self.encoder(x)  # Get hidden state
        # Decode
        h_n_repeated = h_n.repeat(x.size(1), 1, 1).permute(1, 0, 2)  # Repeat latent vector
        x_reconstructed, _ = self.decoder(h_n_repeated)
        return x_reconstructed

# Step 2: Generate the Synthetic Data
def generate_synthetic_data(n_seq=10000, seq_len=50):
    data = np.random.rand(n_seq, seq_len)
    for seq in data:
        i = np.random.randint(20, 31)
        seq[i-5:i+5] *= 0.1
    return np.clip(data, 0, 1)

# Step 3: Train the LSTM Autoencoder
def train_model(model, dataloader, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_reconstructed = model(x)
            loss = criterion(x_reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")

# Step 4: Plot Original and Reconstructed Sequences
def plot_sequences(original, reconstructed, title):
    plt.figure(figsize=(10, 6))
    for i in range(3):  # Plot 3 sequences
        plt.plot(original[i], label=f"Original {i+1}")
        plt.plot(reconstructed[i], label=f"Reconstructed {i+1}")
    plt.legend()
    plt.title(title)
    plt.show()

# Main Script
if __name__ == "__main__":
    # Hyperparameters
    input_size = 1  # Each value in the sequence is a single feature
    hidden_size = 64
    batch_size = 128
    learning_rate = 1e-3
    epochs = 100

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate and prepare data
    data = generate_synthetic_data()
    data = data[..., np.newaxis]  # Add feature dimension for PyTorch (n_seq, seq_len, input_size)
    train_size = int(0.6 * len(data))
    val_size = int(0.2 * len(data))
    test_size = len(data) - train_size - val_size
    train_data, val_data, test_data = np.split(data, [train_size, train_size + val_size])

    # Convert to PyTorch tensors
    train_tensor = torch.tensor(train_data, dtype=torch.float32)
    val_tensor = torch.tensor(val_data, dtype=torch.float32)
    test_tensor = torch.tensor(test_data, dtype=torch.float32)

    # Create DataLoader
    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, and loss function
    model = LSTMAutoencoder(input_size=input_size, hidden_size=hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Train the model
    train_model(model, train_loader, optimizer, criterion, epochs)

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        test_reconstructed = model(test_tensor.to(device)).cpu().numpy()
        plot_sequences(test_tensor.squeeze(-1).numpy()[:3], test_reconstructed.squeeze(-1)[:3], "Original vs Reconstructed")