import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll.base import scope

from src.models import CNNModel
from .sliding_window import ECGDataset

WINDOW_SIZE = 100
STRIDE = 10
EPOCHS = 2


class HyperoptSearch:

    def __init__(self, X: np.ndarray, y: pd.DataFrame):
        self._X_train = X
        self._y_train = y

        self._search_space = {
            'batch_size': scope.int(hp.quniform('batch_size', 16, 64, 8)),
            'learning_rate': hp.loguniform('learning_rate', np.log(1e-4), np.log(1e-2)),
            'dropout_rate': hp.uniform('dropout_rate', 0.3, 0.7),
            'conv_filters': hp.choice('conv_filters', [(32, 64, 128), (64, 128, 256)])
        }

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trials = Trials()
        best_params = fmin(fn=self.objective, space=self._search_space, algo=tpe.suggest, max_evals=20, trials=trials)

        print(f"Best parameters found: {best_params}")
        # best_params['conv_filters'] = [(32, 64, 128), (64, 128, 256)][best_params['conv_filters']]

    def cross_val_score_model(self, params, X, y, n_splits=3):
        batch_size = int(params['batch_size'])
        learning_rate = params['learning_rate']
        dropout_rate = params['dropout_rate']
        conv_filters = params['conv_filters']

        kf = KFold(n_splits=n_splits)
        val_losses = []

        for train_idx, val_idx in kf.split(X):
            train_dataset = ECGDataset(
                X[train_idx], y.iloc[train_idx]["ground_truth"].values, window_size=WINDOW_SIZE, stride=STRIDE
            )
            val_dataset = ECGDataset(
                X[val_idx], y.iloc[val_idx]["ground_truth"].values, window_size=WINDOW_SIZE, stride=STRIDE
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = CNNModel(
                input_channels=12,
                num_classes=5,
                conv_filters=conv_filters,
                dropout_rate=dropout_rate
            ).to(self._device)

            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            model.train()
            for epoch in range(EPOCHS):
                for data, labels in train_loader:
                    data = data.permute(0, 2, 1).to(self._device)  # Change shape to (batch_size, channels, time_steps)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, labels in val_loader:
                    data = data.permute(0, 2, 1).to(self._device)
                    labels = labels.to(self._device)

                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

            val_losses.append(val_loss / len(val_loader))

        return np.mean(val_losses)

    def objective(self, params):
        print(f"Training with params: {params}")
        loss = self.cross_val_score_model(params, self._X_train, self._y_train, n_splits=3)
        return loss


if __name__ == '__main__':
    X = np.load("../../data/processed/X.npy")
