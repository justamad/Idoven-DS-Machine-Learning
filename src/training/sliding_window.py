import torch
import numpy as np

from torch.utils.data import Dataset


def calculate_sliding_window_indices(data: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    num_windows = (data.shape[0] - window_size) // stride + 1
    windows = np.array([data[i * stride: i * stride + window_size, :] for i in range(num_windows)])
    return windows


class ECGDataset(Dataset):

    def __init__(self, ecg_data: np.ndarray, labels: np.ndarray, window_size: int, stride: int):
        self._ecg_data = ecg_data
        self._labels = labels
        self._window_size = window_size
        self._stride = stride

        self._windows = [calculate_sliding_window_indices(x, self._window_size, self._stride) for x in self._ecg_data]
        self._windows = np.concatenate(self._windows, axis=0)  # Stack all windows
        self._labels = np.repeat(
            self._labels,
            self._windows.shape[0] // len(self._labels)
        )

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        window = self._windows[idx]
        label = self._labels[idx]
        window = torch.tensor(window, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return window, label
