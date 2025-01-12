from random import random

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import optim, nn
from torch.utils.data import Dataset, DataLoader

from lstm_autoencoder import LSTMAutoEncoder
from lstm_autoencoder_predictor import LSTMAutoencoderPredictor

stocks = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
features = ['open', 'high', 'low', 'close', 'volume']


def create_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i+seq_length][features].values
        sequences.append(seq)
    return sequences


def create_sequences_with_targets(data, seq_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length):
        seq = data.iloc[i:i + seq_length][features].values
        target = data.iloc[i+1:i + seq_length+1][features].values  # xt+1
        sequences.append(seq)
        targets.append(target)
    return sequences, targets

def section_1():
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

def section_2():
    data = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
    features = ['open', 'high', 'low', 'close', 'volume']
    df = data.sort_values(['symbol', 'date']).reset_index(drop=True)
    df = df.dropna()
    scaler = MinMaxScaler()
    df_scaled = df.copy()

    for symbol, group in df.groupby('symbol'):
        scaler = MinMaxScaler()
        df_scaled.loc[group.index, features] = scaler.fit_transform(group[features])

    all_sequences = []
    all_symbols = []
    all_dates = []

    unique_symbols = df_scaled['symbol'].unique()
    SEQ_LENGTH = 30
    #unique_symbols = ["AAPL", "MSFT", "AMZN", "SLG", "GOOGL", "PCG", "HAL", "RL", "FIS", "XEC", "UPS", "LLL", "KMI", "AEP", "ALK", "AVGO"]
    for symbol in unique_symbols:
        print("Creating sequences for symbol {}".format(symbol))
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences = create_sequences(symbol_data, seq_length=SEQ_LENGTH)
        all_sequences.extend(sequences)
        all_symbols.extend([symbol] * len(sequences))
        all_dates.extend(symbol_data['date'][SEQ_LENGTH - 1:,].values)

    print("Finished creating sequences for all symbols")
    dataset = StockDataset(all_sequences)
    BATCH_SIZE = 64
    KFOLDS = 3
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    INPUT_SIZE = len(features)
    HIDDEN_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 1e-3

    model = LSTMAutoEncoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    fold_results = {}
    best_val_loss = float('inf')
    best_model_state = None

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{KFOLDS}')

        # Split the data
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize a new model for each fold
        fold_model = LSTMAutoEncoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
        fold_model.train()

        # Define optimizer and loss for this fold
        fold_optimizer = torch.optim.Adam(fold_model.parameters(), lr=LEARNING_RATE)
        fold_criterion = nn.MSELoss()

        # Training loop
        for epoch in range(EPOCHS):
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
                print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {epoch_loss:.4f}')

        # Validation
        fold_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, _ = fold_model(batch)
                loss = fold_criterion(reconstructed, batch)
                val_loss += (loss.item() / len(val_loader))
        print(f'Validation Loss for fold {fold + 1}: {val_loss:.4f}\n')

        fold_results[fold] = val_loss

        # Check if this fold has the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = fold_model.state_dict()

    # Load the best model state
    best_model = LSTMAutoEncoder(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    best_model.load_state_dict(best_model_state)
    torch.save(best_model_state, f"output/sp500/models/best_model_section_2.pth")
    best_model.eval()

    print(f'Best Validation Loss: {best_val_loss:.4f}')
    selected_symbols = ["AAPL", "MSFT", "AMZN", "SLG", "GOOGL"]
    print(f'Selected Symbols: {selected_symbols}')

    for symbol in selected_symbols:
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences = create_sequences(symbol_data, SEQ_LENGTH)
        dates = symbol_data['date'][SEQ_LENGTH - 1:].values
        original_sequences = np.array(sequences)

        # Convert to tensor
        sequences_tensor = torch.tensor(sequences, dtype=torch.float32).to(device)

        # Reconstruct
        with torch.no_grad():
            reconstructed, _ = best_model(sequences_tensor)

        reconstructed = reconstructed.cpu().numpy()

        # Inverse transform to get actual values
        # Note: Since scaling was done per symbol, we need to inverse transform accordingly
        # We'll use the scaler fitted earlier
        original = scaler.inverse_transform(original_sequences.reshape(-1, len(features)))
        reconstructed = scaler.inverse_transform(reconstructed.reshape(-1, len(features)))

        # Reshape back to sequences
        original = original.reshape(-1, SEQ_LENGTH, len(features))
        reconstructed = reconstructed.reshape(-1, SEQ_LENGTH, len(features))

        # For visualization, we'll plot the 'close' price of the last time step in each sequence
        original_close = original[:, -1, features.index('close')]
        reconstructed_close = reconstructed[:, -1, features.index('close')]

        # Plotting
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

def section_3():
    data = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
    features = ['open', 'high', 'low', 'close', 'volume']
    df = data.sort_values(['symbol', 'date']).reset_index(drop=True)
    df = df.dropna()
    df_scaled = df.copy()

    for symbol, group in df.groupby('symbol'):
        scaler = MinMaxScaler()
        df_scaled.loc[group.index, features] = scaler.fit_transform(group[features])

    SEQ_LENGTH = 30  # Number of time steps in each sequence

    all_sequences = []
    all_targets = []
    all_symbols = []
    all_dates = []

    unique_symbols = df_scaled['symbol'].unique()
    # unique_symbols = ["AAPL", "MSFT", "AMZN"]

    for symbol in unique_symbols:
        print("Creating targets and sequences for symbol {}".format(symbol))
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences, targets = create_sequences_with_targets(symbol_data, SEQ_LENGTH)
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        all_symbols.extend([symbol] * len(sequences))
        all_dates.extend(symbol_data['date'][SEQ_LENGTH:].values)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StockDatasetPredictor(all_sequences, all_targets)


    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_SIZE = len(features)  # 5 features
    HIDDEN_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    # Initialize lists to store losses for visualization
    train_recon_losses = []
    train_pred_losses = []
    val_recon_losses = []
    val_pred_losses = []

    # Initialize variables to track the best model
    best_val_loss = float('inf')
    best_model_state = None
    best_model_train_recon_losses = []
    best_model_train_pred_losses = []

    KFOLDS = 3
    kfold = KFold(n_splits=KFOLDS, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f'Fold {fold + 1}/{KFOLDS}')

        # Split the data
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)

        # Create DataLoaders
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

        # Initialize the model for this fold
        model = LSTMAutoencoderPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
        model.train()

        # Define loss functions and optimizer
        reconstruction_criterion = nn.MSELoss()
        prediction_criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        train_recon_losses = []
        train_pred_losses = []
        val_recon_losses = []
        val_pred_losses = []
        # Training epochs
        for epoch in range(NUM_EPOCHS):
            epoch_recon_loss = 0.0
            epoch_pred_loss = 0.0

            batch_idx = 0
            for batch in train_loader:
                batch_idx += 1
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()

                # Forward pass
                reconstructed, predicted = model(sequences, targets)

                # Calculate losses
                recon_loss = reconstruction_criterion(reconstructed, sequences)
                pred_loss = prediction_criterion(predicted, targets)

                # Total loss
                total_loss = recon_loss + pred_loss

                # Backward pass and optimization
                total_loss.backward()

                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                # Accumulate losses
                epoch_recon_loss += recon_loss.item()
                epoch_pred_loss += pred_loss.item()
                print(f"epoch {epoch + 1} batch {batch_idx} - recon_loss: {epoch_recon_loss}, pred_loss: {epoch_pred_loss}")
            # Average losses for the epoch
            avg_recon_loss = epoch_recon_loss / len(train_loader)
            avg_pred_loss = epoch_pred_loss / len(train_loader)

            train_recon_losses.append(avg_recon_loss)
            train_pred_losses.append(avg_pred_loss)

            if (epoch + 1) % 10 == 0:
                print(
                    f'Epoch [{epoch + 1}/{NUM_EPOCHS}], Reconstruction Loss: {avg_recon_loss:.4f}, Prediction Loss: {avg_pred_loss:.4f}')

        # Validation
        model.eval()
        val_recon_loss = 0.0
        val_pred_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                sequences, targets = batch
                sequences = sequences.to(device)
                targets = targets.to(device)

                # Forward pass
                reconstructed, predicted = model(sequences, targets)

                # Calculate losses
                recon_loss = reconstruction_criterion(reconstructed, sequences)
                pred_loss = prediction_criterion(predicted, targets)

                # Accumulate validation losses
                val_recon_loss += recon_loss.item()
                val_pred_loss += pred_loss.item()

        # Average validation losses
        avg_val_recon_loss = val_recon_loss / len(val_loader)
        avg_val_pred_loss = val_pred_loss / len(val_loader)

        val_recon_losses.append(avg_val_recon_loss)
        val_pred_losses.append(avg_val_pred_loss)

        print(f'Validation Reconstruction Loss for fold {fold + 1}: {avg_val_recon_loss:.4f}')
        print(f'Validation Prediction Loss for fold {fold + 1}: {avg_val_pred_loss:.4f}\n')

        # Total validation loss for best model selection
        total_val_loss = avg_val_recon_loss + avg_val_pred_loss
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            best_model_state = model.state_dict()
            best_model_train_recon_losses = train_recon_losses
            best_model_train_pred_losses = train_pred_losses
            best_model_val_recon_losses = val_recon_losses
            best_model_val_pred_losses = val_pred_losses

    best_model = LSTMAutoencoderPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE).to(device)
    best_model.load_state_dict(best_model_state)
    torch.save(best_model.state_dict(), 'output/sp500/models/best_model_section_3.pth')
    best_model.eval()

    print(f'Best Validation Total Loss: {best_val_loss:.4f}')

    # Plot Training Reconstruction Loss
    plt.figure(figsize=(12, 6))

    # First subplot: Training Reconstruction Loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
    plt.plot(range(1, NUM_EPOCHS + 1), best_model_train_recon_losses, label='Training Reconstruction Loss',
             color='blue')
    plt.title('Training Reconstruction Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # Second subplot: Training Prediction Loss
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
    plt.plot(range(1, NUM_EPOCHS + 1), best_model_train_pred_losses, label='Training Prediction Loss', color='orange')
    plt.title('Training Prediction Loss vs. Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)

    # Adjust layout and show the combined plot
    plt.tight_layout()
    plt.savefig('output/sp500/section3/best_model_losses.png')
    plt.show()

    # Selecting three random symbols for visualization
    selected_symbols = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    print(f'Selected Symbols: {selected_symbols}')

    for symbol in selected_symbols:
        # Extract scaled data for the selected symbol
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences, targets = create_sequences_with_targets(symbol_data, SEQ_LENGTH)
        dates = symbol_data['date'][SEQ_LENGTH:].values
        original_sequences = np.array(sequences)
        original_targets = np.array(targets)

        # Convert to tensor
        sequences_tensor = torch.tensor(original_sequences, dtype=torch.float32).to(device)
        targets_tensor = torch.tensor(original_targets, dtype=torch.float32).to(device)
        # Reconstruct and Predict using the best model
        with torch.no_grad():
            reconstructed, predicted = best_model(sequences_tensor, targets_tensor)

        # Move tensors to CPU and convert to NumPy arrays
        reconstructed = reconstructed.cpu().numpy()
        predicted = predicted.cpu().numpy()

        # Inverse transform to get actual values
        scaler_symbol = MinMaxScaler()
        original_data = df[df['symbol'] == symbol][features]
        scaler_symbol.fit(original_data)

        # Inverse transform sequences, reconstructions, and predictions
        original = scaler_symbol.inverse_transform(original_sequences.reshape(-1, len(features))).reshape(-1, SEQ_LENGTH, len(features))
        reconstructed = scaler_symbol.inverse_transform(reconstructed.reshape(-1, len(features))).reshape(-1, SEQ_LENGTH, len(features))
        predicted = scaler_symbol.inverse_transform(predicted.reshape(-1, len(features))).reshape(-1, SEQ_LENGTH, len(features))

        # Extract 'close' prices
        original_close = original[:, -1, features.index('close')]
        reconstructed_close = reconstructed[:, -1, features.index('close')]
        predicted_close = predicted[:, -1, features.index('close')]

        # Plot Original vs Reconstructed Close Price
        plt.figure(figsize=(12, 6))
        plt.plot(dates, original_close, label='Original Close Price')
        plt.plot(dates, reconstructed_close, label='Reconstructed Close Price', alpha=0.7)
        plt.title(f'Original vs Reconstructed Close Price for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/sp500/section3/{symbol}-reconstruction_vs_original.png')
        plt.show()

        # Plot Original Close Price and Predicted Close Price
        plt.figure(figsize=(12, 6))
        plt.plot(dates, original_close, label='Original Close Price (t)')
        plt.plot(dates, predicted_close, label='Predicted Close Price (t+1)', color='green', alpha=0.7)
        plt.title(f'Original Close Price and Predicted Close Price for {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'output/sp500/section3/{symbol}-prediction_vs_original.png')
        plt.show()


def multi_step_predict(model, input_sequence, prediction_steps, device):
    """
    Perform multi-step prediction using the trained model.

    Parameters:
    - model: Trained LSTMAutoencoderPredictor model.
    - input_sequence: Tensor of shape (1, seq_length, input_size).
    - prediction_steps: Number of future steps to predict.
    - device: Device to perform computations on.

    Returns:
    - predictions: List of predicted steps.
    """
    model.eval()
    predictions = torch.empty((1, 0, 5))

    current_input = input_sequence.to(device)  # Shape: (1, seq_length, input_size)

    with torch.no_grad():
        for _ in range(prediction_steps):
            # Get the prediction for the next step
            predicted = model(current_input)  # Shape: (1, input_size)
            predicted_next_input = predicted[:, -1, :].unsqueeze(1)
            # Append the prediction to the list
            predictions = torch.cat((predictions, predicted_next_input), dim = 1)

            # Prepare the next input by appending the prediction and removing the first time step
            current_input = torch.cat((current_input[:, 1:, :], predicted_next_input), dim=1)  # Shape: (1, seq_length, input_size)

    return predictions  # Shape: (prediction_steps, input_size)


def one_step_predict(model, input_sequence, device):
    """
    Perform one-step prediction using the trained model.

    Parameters:
    - model: Trained LSTMAutoencoderPredictor model.
    - input_sequence: Tensor of shape (1, seq_length, input_size).
    - device: Device to perform computations on.

    Returns:
    - prediction: Predicted next step as a NumPy array.
    """
    model.eval()

    with torch.no_grad():
        prediction = model(input_sequence.to(device))  # Shape: (1, input_size)
        prediction = prediction[:, -1, :].unsqueeze(1)

    return prediction

def section_4():
    data = pd.read_csv('data/sp-500-stock-prices/SP 500 Stock Prices 2014-2017.csv', parse_dates=['date'])
    features = ['open', 'high', 'low', 'close', 'volume']
    df = data.sort_values(['symbol', 'date']).reset_index(drop=True)

    df_scaled = df.copy()

    for symbol, group in df.groupby('symbol'):
        scaler = MinMaxScaler()
        df_scaled.loc[group.index, features] = scaler.fit_transform(group[features])

    SEQ_LENGTH = 30  # Number of time steps in each sequence

    all_sequences = []
    all_targets = []
    all_symbols = []
    all_dates = []

    unique_symbols = df_scaled['symbol'].unique()
    unique_symbols = ["AAPL", "MSFT", "AMZN"]

    for symbol in unique_symbols:
        print("Creating targets and sequences for symbol {}".format(symbol))
        symbol_data = df_scaled[df_scaled['symbol'] == symbol].reset_index(drop=True)
        sequences, targets = create_sequences_with_targets(symbol_data, SEQ_LENGTH)
        all_sequences.extend(sequences)
        all_targets.extend(targets)
        all_symbols.extend([symbol] * len(sequences))
        all_dates.extend(symbol_data['date'][SEQ_LENGTH:].values)

    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = StockDatasetPredictor(all_sequences, all_targets)
    train_val_size = 0.8
    test_size = 0.2

    train_val_indices, test_indices = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        shuffle=False  # Ensuring temporal consistency
    )

    # Now, split training+validation into training and validation
    train_size = 0.75  # 75% of 80% = 60% of total data
    val_size = 0.25  # 25% of 80% = 20% of total data

    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        shuffle=False  # Ensuring temporal consistency within training+validation
    )

    # Create subsets
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)

    # Create DataLoaders
    BATCH_SIZE = 64

    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=1, shuffle=False)  # Batch size 1 for sequential prediction


    # Hyperparameters
    BATCH_SIZE = 64
    INPUT_SIZE = len(features)  # 5 features
    HIDDEN_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-3

    model = LSTMAutoencoderPredictor(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE)
    model.load_state_dict(torch.load('output/sp500/models/best_model_section_3.pth', weights_only=True))

    # Initialize lists to store evaluation metrics
    multi_step_mse = []
    multi_step_mae = []
    one_step_mse = []
    one_step_mae = []

    # Define number of steps to predict
    PREDICTION_STEPS = 10  # Adjust based on your requirements

    for batch in test_loader:
        sequences, targets = batch  # sequences: (1, seq_length, input_size), targets: (1, seq_length, input_size)
        sequences = sequences.to(device)


        # Split the sequence into input and actual target
        T = sequences.size(1)
        half_T = T // 2

        input_seq = sequences[:, :half_T, :]  # First half
        actual_targets = sequences[:, half_T:, :]  # Second half

        # Multi-Step Prediction
        predicted_steps = multi_step_predict(model, input_seq, prediction_steps=half_T, device=device)

        # Rescale predictions and actual targets if necessary
        # Assuming scaler was fitted per symbol, adjust accordingly
        # For simplicity, assuming scaler is available for the current symbol
        # Replace 'scaler' with the appropriate scaler per symbol
        # Here, assuming a single scaler for demonstration
        scaler = MinMaxScaler()
        scaler.fit(df[['open', 'high', 'low', 'close', 'volume']])  # Fit on entire data or per symbol as needed

        mse = nn.MSELoss()

        actual_targets_rescaled = scaler.inverse_transform(
            actual_targets.cpu().numpy().reshape(-1, len(features))).reshape(-1, len(features))
        predicted_steps_rescaled = scaler.inverse_transform(predicted_steps.cpu().numpy().reshape(-1, len(features))).reshape(-1, len(features))

        # Compute Metrics for Multi-Step Prediction
        multi_step_mse.append(mse(predicted_steps, actual_targets).item())

        # One-Step Prediction
        prediction = torch.empty(1, 0, 5)
        for _ in range(half_T):
            one_step_prediction = one_step_predict()
        one_step_pred = one_step_predict(model, input_seq, device=device)
        one_step_mse.append(mse(one_step_pred, actual_targets).item())
        one_step_pred_rescaled = scaler.inverse_transform(one_step_pred.reshape(1, -1).flatten(axis=0))
        actual_one_step = scaler.inverse_transform(actual_targets[:, 0, :].cpu().numpy())

        # Compute Metrics for One-Step Prediction
        mse_one = mean_squared_error(actual_one_step, one_step_pred_rescaled)
        mae_one = mean_absolute_error(actual_one_step, one_step_pred_rescaled)
        one_step_mse.append(mse_one)
        one_step_mae.append(mae_one)

    # Calculate average metrics
    avg_multi_step_mse = np.mean(multi_step_mse)
    avg_multi_step_mae = np.mean(multi_step_mae)
    avg_one_step_mse = np.mean(one_step_mse)
    avg_one_step_mae = np.mean(one_step_mae)

    print("=== Multi-Step Prediction Metrics ===")
    print(f"Average MSE: {avg_multi_step_mse:.4f}")
    print(f"Average MAE: {avg_multi_step_mae:.4f}")

    print("\n=== One-Step Prediction Metrics ===")
    print(f"Average MSE: {avg_one_step_mse:.4f}")
    print(f"Average MAE: {avg_one_step_mae:.4f}")


if __name__ == '__main__':
    seq_length = 30  # Example sequence length
    hidden_size = 128
    epochs = 50
    batch_size = 64
    section_3()
    #section_4()
    # print(stocks.head())
    #section_2()
