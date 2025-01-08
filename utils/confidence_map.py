import torch
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def compute_confidence_map(signal, device, config):
    
    distance = config["confidence"]["distance"]
    sigma = config["confidence"]["sigma"]
    signal = signal.squeeze().detach().cpu().numpy()
    energy = np.mean(signal ** 2)

    adaptive_height = energy
    adaptive_distance = int(len(signal) / distance)

    peaks, properties = find_peaks(abs(signal), height=adaptive_height, distance=adaptive_distance)
    confidence_map = np.zeros_like(signal)
    confidence_map[peaks] = properties['peak_heights']

    smoothed_confidence_map = gaussian_filter1d(confidence_map, sigma=sigma)
    
    return torch.tensor(smoothed_confidence_map, device=device).unsqueeze(0)

