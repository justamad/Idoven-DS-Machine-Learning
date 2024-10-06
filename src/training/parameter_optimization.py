import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
import json

from typing import List, Dict
from torch.utils.data import DataLoader
from os.path import join
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical

from src.models import CNNModel
from src.training import ECGDataset
from src.data_loading import prepare_ground_truth_data


class HyperparameterSearch:

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: pd.DataFrame,
        X_test: np.ndarray,
        y_test: pd.DataFrame,
        model_path: str,
        win_size: int,
        stride: int,
        epochs: int,
    ):
        self._X_train = X_train
        self._y_train = y_train

        self._X_test = X_test
        self._y_test = y_test

        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._model_path = model_path

        self._win_size = win_size
        self._stride = stride
        self._epochs = epochs

    def perform_search(self, n_calls: int):
        search_space = [
            Integer(8, 16, name='batch_size'),
            Real(1e-4, 1e-2, name='learning_rate', prior='log-uniform'),
            Real(0.3, 0.7, name='dropout_rate'),
            Categorical([(16, 32, 64), (32, 64, 128)], name='conv_filters')
        ]

        result = gp_minimize(self.objective, search_space, n_calls=n_calls, random_state=42)
        print(f"Best hyperparameters found: {result.x}")

        os.makedirs(self._model_path, exist_ok=True)

        with open(join(self._model_path, "best_params.pkl"), 'wb') as file:
            pickle.dump(result.x, file)

    def objective(self, params: List):
        print(f"Training with params: {params}")
        loss = self.train_model_with_k_fold(params, n_splits=3)
        return loss

    def train_model_with_k_fold(self, params: List, n_splits: int=3):
        batch_size, learning_rate, dropout_rate, conv_filters = params
        batch_size = int(batch_size)

        kf = KFold(n_splits=n_splits)
        val_losses = []

        for train_idx, val_idx in kf.split(self._X_train):
            train_dataset = ECGDataset(
                self._X_train[train_idx],
                self._y_train.iloc[train_idx]["ground_truth"].values,
                window_size=self._win_size,
                stride=self._stride,
            )
            val_dataset = ECGDataset(
                self._X_train[val_idx],
                self._y_train.iloc[val_idx]["ground_truth"].values,
                window_size=self._win_size,
                stride=self._stride,
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

            model = CNNModel(input_channels=12, conv_filters=conv_filters, dropout_rate=dropout_rate).to(self._device)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

            train_loss_history = []
            val_loss_history = []
            model.train()
            for epoch in range(self._epochs):
                running_loss = 0.0
                for data, labels in train_loader:
                    data = data.permute(0, 2, 1).to(self._device)  # Change shape to (batch_size, channels, time_steps)
                    labels = labels.to(self._device)

                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                # Calculate average loss for the epoch
                epoch_loss = running_loss / len(train_loader)
                train_loss_history.append(epoch_loss)  # Save epoch loss

                # Validation loop after each epoch
                model.eval()  # Set model to evaluation mode
                val_running_loss = 0.0
                with torch.no_grad():  # Disable gradient computation for validation
                    for val_data, val_labels in val_loader:
                        val_data = val_data.permute(0, 2, 1).to(self._device)
                        val_labels = val_labels.to(self._device)

                        val_outputs = model(val_data)
                        val_loss = criterion(val_outputs, val_labels)
                        val_running_loss += val_loss.item()

                # Calculate average validation loss for the epoch
                val_epoch_loss = val_running_loss / len(val_loader)
                val_loss_history.append(val_epoch_loss)

            val_losses.append(val_epoch_loss)

        return np.mean(val_losses)

    def evaluate_model(self, params: List):
        batch_size, learning_rate, dropout_rate, conv_filters = params
        batch_size = int(batch_size)

        test_dataset = ECGDataset(
            self._X_test,
            self._y_test["ground_truth"].values,
            window_size=self._win_size,
            stride=self._stride,
        )
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        train_dataset = ECGDataset(
            self._X_train,
            self._y_train["ground_truth"].values,
            window_size=self._win_size,
            stride=self._stride,
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model = CNNModel(input_channels=12, conv_filters=conv_filters, dropout_rate=dropout_rate).to(self._device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Retrain the model with best parameter combination on all train data
        train_loss_history = []
        model.train()
        for epoch in range(self._epochs):
            running_loss = 0.0
            for data, labels in train_loader:
                data = data.permute(0, 2, 1).to(self._device)  # Change shape to (batch_size, channels, time_steps)
                labels = labels.to(self._device)

                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            train_loss_history.append(epoch_loss)  # Save epoch loss

        print(f"Train loss: {train_loss_history[-1]}")

        # Evaluate entire model on Test data
        model.eval()  # Set model to evaluation mode
        test_loss_history = []
        test_running_loss = 0.0
        labels, predictions = [], []
        with torch.no_grad():  # Disable gradient computation for validation
            for test_data, test_labels in test_loader:
                test_data = test_data.permute(0, 2, 1).to(self._device)
                test_labels = test_labels.to(self._device)

                test_outputs = model(test_data)
                prediction = model.predict(test_outputs)

                test_loss = criterion(test_outputs, test_labels)
                test_running_loss += test_loss.item()

                labels.append(test_labels)
                predictions.append(prediction)

            epoch_loss = test_running_loss / len(train_loader)
            test_loss_history.append(epoch_loss)

        labels_flat = torch.cat(labels).view(-1).numpy()
        predictions_flat = torch.cat(predictions).view(-1).numpy()
        return labels_flat, predictions_flat



if __name__ == '__main__':
    X = np.load("../../data/processed/X.npy")
    Y = pd.read_csv("../../data/processed/y.csv")
    Y['diagnostic_superclass'] = Y['diagnostic_superclass'].apply(json.loads)

    Y = prepare_ground_truth_data(Y)

    test_fold = 10
    X_train = X[np.where(Y.strat_fold != test_fold)]
    y_train = Y[(Y.strat_fold != test_fold)]
    X_test = X[np.where(Y.strat_fold == test_fold)]
    y_test = Y[Y.strat_fold == test_fold]

    model_path = "../../param_search"

    win_size = 100
    stride = 100
    epochs = 1
    opt_search = HyperparameterSearch(X_train, y_train, X_test, y_test, model_path, win_size, stride, epochs)
    # opt_search.perform_search(n_calls=10)

    with open(join(model_path, "best_params.pkl"), 'rb') as file:
        loaded_results = pickle.load(file)

    print(loaded_results)
    predictions, labels = opt_search.evaluate_model(loaded_results)
    print(predictions)
    print(labels)
