import argparse
from itertools import product
import os
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader

from lstm_autoencoder import LSTMAutoEncoder
from synthetic_data_utils import generate_synthetic_data, plot_examples
from utils import save_grid_search_results_to_csv, split_data_to_train_validation_test, get_optimizer


def train_and_evaluate_lstm_ae(train_tensor_dataset, validation_tensor_dataset, device, hyperparameters):
    hidden_size = hyperparameters['hidden_size']
    learning_rate = hyperparameters['learning_rate']
    gradient_clipping = hyperparameters['gradient_clipping']
    input_size = hyperparameters['input_size']
    epochs = hyperparameters['epochs']
    batch_size = hyperparameters['batch_size']

    train_loader = DataLoader(train_tensor_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_tensor_dataset, batch_size=batch_size, shuffle=True)

    model = LSTMAutoEncoder(input_size=input_size, hidden_size=hidden_size)
    optimizer = get_optimizer(model, args)
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for batch_idx, inputs in enumerate(train_loader):
            outputs, _ = model(inputs)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            total_loss += loss.item()
        scheduler.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in validation_loader:
            inputs = batch.to(device)
            outputs = model(inputs)
            val_loss += criterion(outputs, inputs).item()

    val_loss /= len(validation_loader)

    return model, val_loss

def grid_search(train_tensor, validation_tensor, device):
    hyperparameters_options = {
        "hidden_size": [16, 32, 64],
        "learning_rate": [1e-3, 5e-3, 1e-2],
        "gradient_clipping": [0.5, 1.0, 5.0],
        "input_size": [1],
        "epochs": [150],
        "batch_size": [128],
    }

    keys = hyperparameters_options.keys()
    combinations = product(*hyperparameters_options.values())
    hyperparameter_combinations = [dict(zip(keys, values)) for values in combinations]


    train_tensor_dataset = TensorDataset(train_tensor)
    validation_tensor_dataset = TensorDataset(validation_tensor)

    results = []

    for hyperparameters in hyperparameter_combinations:
        model, val_loss = train_and_evaluate_lstm_ae(train_tensor_dataset, validation_tensor_dataset, device, hyperparameters)
        results.append({"hyperparameters": hyperparameters, "val_loss": val_loss})
        print(f"Evaluated {hyperparameters}: Validation Loss = {val_loss:.6f}")
        filename = f"model_hidden{hyperparameters['hidden_size']}_lr{hyperparameters['learning_rate']}_clip{hyperparameters['gradient_clipping']}.pth"
        filepath = os.path.join("output/synthetic_data/models", filename)
        torch.save({"model_state_dict": model.state_dict(),
                    "hyperparameters": hyperparameters,
                    "val_loss": val_loss}, filepath)

    save_grid_search_results_to_csv(
        [{"hidden_size": h["hyperparameters"]["hidden_size"], "learning_rate": h["hyperparameters"]["learning_rate"], "gradient_clipping": h["hyperparameters"]["gradient_clipping"], "val_loss": h["val_loss"]} for h in results]
    )

def reconstruct_input(model, x):
    model.eval()
    with torch.no_grad():
        reconstruction, _ = model(x)

    return reconstruction

def plot_original_vs_reconstructed(original, reconstructed):
    time = range(original.shape[0])

    plt.figure(figsize=(10, 6))
    plt.plot(time, original[:, 0], label="Original", alpha=0.8)
    plt.plot(time, reconstructed[:, 0], label="Reconstructed", alpha=0.8, linestyle="--")
    plt.title("Original vs. Reconstructed Sequence")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend()
    plt.grid()
    plt.savefig("./output/synthetic_data/reconstructed.png")
    plt.show()

def section_1a():
    n_seq = 10000
    seq_len = 50
    data = generate_synthetic_data(n_seq=n_seq, seq_len=seq_len)

    plot_examples(data)

def section_1b():
    n_seq = 10000
    seq_len = 50
    data = generate_synthetic_data(n_seq=n_seq, seq_len=seq_len)

    training_data, validation_data, test_data = split_data_to_train_validation_test(data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_tensor = torch.from_numpy(training_data).float().unsqueeze(-1).to(device)
    validation_tensor = torch.from_numpy(validation_data).float().unsqueeze(-1).to(device)
    grid_search(training_tensor, validation_tensor, device)

    # After an over-night run, we got the best results using hidden_size = 64, learning_rate = 0.01, gradient_clipping = 0.5

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=1)
    parser.add_argument("--hidden-size", type = int, default = 32)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-clipping", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--optimizer", type=str, default="adam")

    args = parser.parse_args()

    n_seq = 10000
    seq_len = 50
    data = generate_synthetic_data(n_seq = n_seq, seq_len = seq_len)

    training_data, validation_data, test_data = split_data_to_train_validation_test(data)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_tensor = torch.from_numpy(training_data).float().unsqueeze(-1).to(device)
    validation_tensor = torch.from_numpy(validation_data).float().unsqueeze(-1).to(device)
    args_dict = vars(args)
    model, _ =train_and_evaluate_lstm_ae(training_tensor, validation_tensor, device, args_dict)

    validation_loader = DataLoader(TensorDataset(validation_tensor), batch_size=1, shuffle=True)
    signal, = next(iter(validation_loader))
    reconstructed_signal = reconstruct_input(model, signal)

    plot_original_vs_reconstructed(signal[0].cpu().numpy(), reconstructed_signal[0].cpu().numpy())


