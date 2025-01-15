import argparse

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

from lstm_autoencoder import LSTMAutoEncoder
from lstm_autoencoder_predictor import LSTMAutoencoderPredictor
from utils import get_optimizer

features = ['open', 'high', 'low', 'close', 'volume']

def create_scaled_stocks_dataframe():
    data = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
    features = ['open', 'high', 'low', 'close', 'volume']
    df = data.sort_values(['symbol', 'date']).reset_index(drop=True)
    df = df.dropna()
    df_scaled = df.copy()
    scalers = {}

    for symbol, group in df.groupby('symbol'):
        scaler = MinMaxScaler()
        df_scaled.loc[group.index, features] = scaler.fit_transform(group[features])
        scalers[symbol] = scaler

    return df_scaled, scalers

def create_sequences_for_symbol(symbol_data, seq_length):
    sequences = []
    for i in range(len(symbol_data) - seq_length):
        seq = symbol_data.iloc[i:i + seq_length][features].values
        sequences.append(seq)
    return sequences

def create_sequences(df, seq_length):
    all_sequences = []
    all_symbols = []
    all_dates = []

    unique_symbols = df['symbol'].unique()
    # unique_symbols = ["AAPL", "MSFT", "AMZN", "SLG", "GOOGL", "PCG", "HAL", "RL", "FIS", "XEC", "UPS", "LLL", "KMI", "AEP", "ALK", "AVGO"]
    for symbol in unique_symbols:
        print(f"Creating sequences for symbol {symbol}")
        symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
        sequences = create_sequences_for_symbol(symbol_data, seq_length=seq_length)
        all_sequences.extend(sequences)
        all_symbols.extend([symbol] * len(sequences))
        all_dates.extend(symbol_data['date'][seq_length - 1:, ].values)

    print("Finished creating sequences for all symbols")

    return all_sequences, all_symbols, all_dates

def create_sequences_with_targets_for_symbol(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][features].values
        target = data.iloc[i+1:i + seq_length+1][features].values  # xt+1
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

def create_sequences_with_targets(df, seq_length):
    all_sequences = []
    all_targets = []
    all_symbols = []
    all_dates = []

    unique_symbols = df['symbol'].unique()
    unique_symbols = ["AAPL", "MSFT", "AMZN"]

    for symbol in unique_symbols:
        print("Creating targets and sequences for symbol {}".format(symbol))
        symbol_data = df[df['symbol'] == symbol].reset_index(drop=True)
        sequences, targets = create_sequences_with_targets_for_symbol(symbol_data, seq_length)
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        all_symbols.extend([symbol] * len(sequences))
        all_dates.extend(symbol_data['date'][seq_length:].values)

    return all_sequences, all_targets, all_symbols, all_dates

def calculate_loss(model, criterion, data_loader, device):
    model.eval()
    avg_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            avg_loss += (loss.item() / len(data_loader))

    return avg_loss

def section_1():
    stocks = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
    df_google = stocks[stocks['symbol'] == 'GOOGL']
    df_amzn = stocks[stocks['symbol'] == 'AMZN']

    plt.style.use('default')

    plt.figure(figsize=(14,7))
    plt.plot(df_google['date'], df_google['high'], label='GOOGL', color='blue')
    plt.plot(df_amzn['date'], df_amzn['high'], label='AMZN', color='orange')

    plt.title('Daily High Prices for GOOGL and AMZN', fontsize=20)
    plt.xlabel('Date', fontsize = 16)
    plt.xticks(df_google['date'][::60], rotation=45)

    plt.ylabel('Daily High Price', fontsize = 16)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    plt.savefig('output/sp500/section1.png')
    plt.show()

class StockDataset(Dataset):
    def __init__(self, sequences):
        sequences_array = np.array(sequences)
        self.sequences = torch.tensor(sequences_array, dtype=torch.float32)
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return self.sequences[idx]

def section_2(args):
    seq_length = args.seq_length
    kfolds = args.kfolds
    input_size = args.input_size
    hidden_size = args.hidden_size
    epochs = args.epochs

    df_scaled, scalers = create_scaled_stocks_dataframe()
    all_sequences, all_symbols, all_dates = create_sequences(df_scaled, seq_length)

    dataset = StockDataset(all_sequences)

    batch_size = args.batch_size

    kfold = KFold(n_splits=kfolds, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_results = {}
    best_val_loss = float('inf')
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{kfolds}')

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        fold_model = LSTMAutoEncoder(input_size=input_size, hidden_size=hidden_size).to(device)
        fold_model.train()

        fold_optimizer = get_optimizer(fold_model, args)
        fold_criterion = nn.MSELoss()

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                fold_optimizer.zero_grad()
                reconstructed, _ = fold_model(batch)
                loss = fold_criterion(reconstructed, batch)
                loss.backward()

                fold_optimizer.step()
                epoch_loss += (loss.item() / len(train_loader))
                print(f"Total Epoch Loss After Batch Loss: {epoch_loss}")
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}')

        val_loss = calculate_loss(fold_model, fold_criterion, val_loader, device)
        print(f'Validation Loss for fold {fold + 1}: {val_loss:.4f}\n')

        fold_results[fold] = val_loss

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = fold_model.state_dict()


    best_model = LSTMAutoEncoder(input_size=input_size, hidden_size=hidden_size).to(device)
    best_model.load_state_dict(best_model_state)
    torch.save(best_model_state, f"output/sp500/models/best_model_section_2.pth")
    best_model.eval()

    print(f'Best Validation Loss: {best_val_loss:.4f}')

    selected_symbols = ["AAPL", "MSFT", "AMZN", "SLG", "GOOGL"]
    print(f'Selected Symbols: {selected_symbols}')

    for symbol in selected_symbols:
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences = create_sequences(symbol_data, seq_length)
        dates = symbol_data['date'][seq_length - 1:].values
        original_sequences = np.array(sequences)

        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)
        with torch.no_grad():
            reconstructed, _ = best_model(sequences_tensor)

        reconstructed = reconstructed.cpu().numpy()

        original = scalers[symbol].inverse_transform(original_sequences.reshape(-1, len(features)))
        reconstructed = scalers[symbol].inverse_transform(reconstructed.reshape(-1, len(features)))

        original = original.reshape(-1, seq_length, len(features))
        reconstructed = reconstructed.reshape(-1, seq_length, len(features))

        original_close = original[:, -1, features.index('close')]
        reconstructed_close = reconstructed[:, -1, features.index('close')]

        plt.figure(figsize=(12, 6))
        plt.plot(dates, original_close, label='Original Close Price')
        plt.plot(dates, reconstructed_close, label='Reconstructed Close Price')
        plt.title(f'Original vs Reconstructed Close Price for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.savefig(f'output/sp500/{symbol}-reconstruction.png')
        plt.show()

class StockDatasetPredictor(Dataset):
    def __init__(self, sequences, targets):
        sequences_array = np.array(sequences)
        self.sequences = torch.tensor(sequences_array, dtype=torch.float32)
        targets_array = np.array(targets)
        self.targets = torch.tensor(targets_array, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.targets[idx], dtype=torch.float32)

def section_3(args):
    seq_length = args.seq_length
    kfolds = args.kfolds
    input_size = args.input_size
    hidden_size = args.hidden_size # 64
    epochs = args.epochs # 50
    batch_size = args.batch_size # 64

    df_scaled, scalers = create_scaled_stocks_dataframe()
    all_sequences, all_targets, all_symbols, all_dates = create_sequences_with_targets(df_scaled, seq_length)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StockDatasetPredictor(all_sequences, all_targets)

    best_val_loss = float('inf')
    best_model_state = None
    best_model_train_recon_losses = []
    best_model_train_pred_losses = []

    kfold = KFold(n_splits=kfolds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{kfolds}')

        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = LSTMAutoencoderPredictor(input_size=input_size, hidden_size=hidden_size).to(device)
        model.train()

        reconstruction_criterion = nn.MSELoss()
        prediction_criterion = nn.MSELoss()
        optimizer = get_optimizer(model, args)

        train_recon_losses, train_pred_losses = [], []
        val_recon_losses, val_pred_losses = [], []

        for epoch in range(epochs):
            epoch_recon_loss = 0.0
            epoch_pred_loss = 0.0

            batch_idx = 0
            for batch in train_loader:
                batch_idx += 1
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                reconstructed, predicted = model(sequences, targets)

                recon_loss = reconstruction_criterion(reconstructed, sequences)
                pred_loss = prediction_criterion(predicted, targets)

                total_loss = recon_loss + pred_loss

                total_loss.backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                epoch_recon_loss += recon_loss.item()
                epoch_pred_loss += pred_loss.item()
                print(f"epoch {epoch + 1} batch {batch_idx} - recon_loss: {epoch_recon_loss}, pred_loss: {epoch_pred_loss}")

            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_pred_loss = epoch_pred_loss / len(train_loader)

            train_recon_losses.append(avg_recon_loss)
            train_pred_losses.append(avg_pred_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{epochs}], Reconstruction Loss: {avg_recon_loss:.4f}, Prediction Loss: {avg_pred_loss:.4f}')

        model.eval()
        val_recon_loss = 0.0
        val_pred_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)

                reconstructed, predicted = model(sequences, targets)

                recon_loss = reconstruction_criterion(reconstructed, sequences)
                pred_loss = prediction_criterion(predicted, targets)

                val_recon_loss += recon_loss.item()
                val_pred_loss += pred_loss.item()

        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_pred_loss = val_pred_loss / len(val_loader)

        val_recon_losses.append(avg_val_recon_loss)
        val_pred_losses.append(avg_val_pred_loss)

        print(f'Validation Reconstruction Loss for fold {fold + 1}: {avg_val_recon_loss:.4f}')
        print(f'Validation Prediction Loss for fold {fold + 1}: {avg_val_pred_loss:.4f}\n')

        total_val_loss = avg_val_recon_loss + avg_val_pred_loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
            best_model_train_recon_losses = train_recon_losses
            best_model_train_pred_losses = train_pred_losses

    best_model = LSTMAutoencoderPredictor(input_size=input_size, hidden_size=hidden_size).to(device)
    best_model.load_state_dict(best_model_state)
    torch.save(best_model.state_dict(), 'output/sp500/models/best_model_section_3.pth')
    best_model.eval()

    print(f'Best Validation Total Loss: {best_val_loss:.4f}')

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), best_model_train_recon_losses, label='Training Reconstruction Loss',
             color='blue')
    plt.title('Training Reconstruction Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), best_model_train_pred_losses, label='Training Prediction Loss', color='orange')
    plt.title('Training Prediction Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('output/sp500/section3/best_model_losses.png')
    plt.show()


def multi_step_predict(model, input_sequence, half_T, device):
    model.eval()
    predictions = torch.empty((1, 0, 5))

    with torch.no_grad():
        for i in range(half_T):
            current_input = input_sequence.unsqueeze(0)
            current_input = current_input[:, i:i + half_T, :].to(device)
            _, predicted = model(current_input)
            predictions = torch.cat((predictions, predicted), dim = 1)

    return predictions


def one_step_predict(model, input_sequence, half_T, device):
    model.eval()
    predictions = torch.empty((1, 0, 5))
    
    with torch.no_grad():
        for i in range(half_T):
            current_input = input_sequence.unsqueeze(0)
            current_input = current_input[:, i - 1 + half_T, :].to(device)
            _, prediction = model(current_input.unsqueeze(0).to(device))  # Shape: (1, input_size)
            predictions = torch.cat((predictions, prediction), dim = 1)

    return predictions


def section_4(args):
    args.seq_length = 200

    seq_length = args.seq_length
    input_size = args.input_size
    hidden_size = args.hidden_size  # 64

    df_scaled, scalers = create_scaled_stocks_dataframe()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = LSTMAutoencoderPredictor(input_size = input_size, hidden_size = hidden_size)
    model.load_state_dict(torch.load('output/sp500/models/best_model_section_3.pth', weights_only=True))

    selected_symbols = ["AAPL", "MSFT", "AMZN", "SLG", "GOOGL"]
    print(f'Selected Symbols: {selected_symbols}')

    for symbol in selected_symbols:
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences = create_sequences_for_symbol(symbol_data, seq_length)
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

        T = seq_length
        half_T = T // 2
        input_tensor = sequences_tensor[0, :T, :].to(device)
        actual_targets = sequences_tensor[0, half_T:, :]
        multi_step_predicted = multi_step_predict(model, input_tensor, half_T, device=device)
        one_step_predicted = one_step_predict(model, input_tensor, half_T, device=device)

        actual_targets_rescaled = scalers[symbol].inverse_transform(
            actual_targets.cpu().numpy().reshape(-1, len(features))).reshape(-1, len(features))
        multi_step_predicted_rescaled = scalers[symbol].inverse_transform(
            multi_step_predicted.cpu().numpy().reshape(-1, len(features))).reshape(-1, len(features))
        one_step_predicted_rescaled = scalers[symbol].inverse_transform(
            one_step_predicted.cpu().numpy().reshape(-1, len(features))).reshape(-1, len(features))


        close_idx = features.index('high')
        actual_close = actual_targets_rescaled[:, close_idx]
        multi_step_close = multi_step_predicted_rescaled[:, close_idx]
        one_step_close = one_step_predicted_rescaled[:, close_idx]

        plt.figure(figsize=(10, 6))
        time_steps = np.arange(1, len(actual_close) + 1)

        plt.plot(time_steps, actual_close, label='Actual Close Price', color='blue')
        plt.plot(time_steps, multi_step_close, label='Multi-Step Prediction', color='orange')
        plt.plot(time_steps, one_step_close, label='One-Step Prediction', color='green')

        plt.title(f'Comparison of Close Prices: Actual vs. Predictions For {symbol}')
        plt.xlabel('Time Step')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/sp500/section4/comparison_of_actual_close_prices-{symbol}.png')
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-size", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gradient-clipping", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--seq-length", type=int, default=30)
    parser.add_argument("--kfolds", type=int, default=3)


    args = parser.parse_args()
    section_4(args)
