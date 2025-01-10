import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_data(n_seq, seq_len):
    data = np.random.rand(n_seq, seq_len)

    for seq in data:
        i = np.random.randint(20, 31)
        seq[i-5:i+5] *= 0.1

    return data

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
