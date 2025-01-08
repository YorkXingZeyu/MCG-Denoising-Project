import matplotlib.pyplot as plt
import os
import torch

def plot_signals(clean_batch, noisy_batch, output, batch_idx, foldername='', filename=None):
    if isinstance(clean_batch, torch.Tensor):
        clean_batch = clean_batch.cpu().detach().numpy()
    if isinstance(noisy_batch, torch.Tensor):
        noisy_batch = noisy_batch.cpu().detach().numpy()
    if isinstance(output, torch.Tensor):
        output = output.cpu().detach().numpy()

    idx = 0  # Visualize the first signal in the batch

    clean_signal = clean_batch[idx, 0, :]
    noisy_signal = noisy_batch[idx, 0, :]
    denoised_signal = output[idx, 0, :]

    plt.figure(figsize=(15, 5))

    plt.subplot(2, 1, 1)
    plt.plot(clean_signal, label="Clean Signal", color='g', linewidth=0.5)
    plt.plot(denoised_signal, label="Denoised Signal", color='b', linewidth=0.5)
    plt.title(f'Clean vs Denoised Signal (Batch {batch_idx}, Index {idx})')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(noisy_signal, label="Noisy Signal", color='r', linewidth=0.5)
    plt.title(f'Noisy Signal (Batch {batch_idx}, Index {idx})')
    plt.legend()

    plt.tight_layout()
    
    filename_folder = os.path.join(foldername, filename)
    os.makedirs(filename_folder, exist_ok=True)

    filepath = os.path.join(filename_folder , f"Batch{batch_idx}_Index{idx}.png")
    plt.savefig(filepath)
    plt.close()
    print(f"Saved plot to {filepath}")
