import numpy as np
import pandas as pd
from torch import optim


def split_data_to_train_validation_test(data, train_percentage = 0.6, validation_percentage = 0.2):
    train_size = int(len(data) * train_percentage)
    validation_size = int(len(data) * validation_percentage)
    train_data, validation_data, test_data = np.split(data, [train_size, train_size + validation_size])

    return train_data, validation_data, test_data

def save_grid_search_results_to_csv(results, filename="output/grid_search_results.csv"):
    df = pd.DataFrame(results)
    df.columns = ["hidden_size", "learning_rate", "gradient_clipping", "val_loss"]
    df.to_csv(filename, index=False)

def get_optimizer(model, args):
    if args.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=args.learning_rate)
    else:
        raise ValueError("Unknown optimizer")
