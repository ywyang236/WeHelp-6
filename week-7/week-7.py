import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def task_1(device):
    print("\n------------------ Task 1 -------------------")
    weight_dataset = WeightDataset("week-6/gender-height-weight.csv")

    train_size = int(0.8 * len(weight_dataset))
    test_size = len(weight_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        weight_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = WeightPredictor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Start training model...")
    train_weight_model(model, train_loader, criterion, optimizer, device)

    print("\nEvaluating model performance...")
    _, scaler_y = weight_dataset.get_scalers()
    mae, mse, percentage_error, z_score_error = evaluate_weight_model(
        model, test_loader, scaler_y, device
    )

    print(f"Average Test Loss (z-score): {z_score_error:.6f}")
    print(f"Mean Squared Error (pounds): {mse:.2f}")
    print(f"Mean Absolute Error (average pounds off): {mae:.2f}")
    print(f"Average Percentage Error: {percentage_error:.2f}%")
    print("-" * 45)


class WeightDataset(Dataset):
    def __init__(self, csv_file):
        dataframe = pd.read_csv(csv_file)
        dataframe["Gender"] = (dataframe["Gender"] == "Male").astype(float)

        input_features = dataframe[["Gender", "Height"]].values
        target_values = dataframe["Weight"].values

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()

        self.input_features = self.scaler_X.fit_transform(input_features)
        self.target_values = self.scaler_y.fit_transform(
            target_values.reshape(-1, 1)
        ).flatten()

        self.input_features = torch.FloatTensor(self.input_features)
        self.target_values = torch.FloatTensor(self.target_values)

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        return self.input_features[idx], self.target_values[idx]

    def get_scalers(self):
        return self.scaler_X, self.scaler_y


class WeightPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=64):
        super(WeightPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.model(x)


def train_weight_model(model, train_loader, criterion, optimizer, device, epochs=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}"
            )


def evaluate_weight_model(model, test_loader, scaler_y, device):
    model.eval()
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X).squeeze().cpu()
            predictions.extend(outputs.numpy())
            actuals.extend(batch_y.numpy())

    z_score_error = np.mean((np.array(predictions) - np.array(actuals)) ** 2)

    predictions = scaler_y.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler_y.inverse_transform(np.array(actuals).reshape(-1, 1))
    mae = np.mean(np.abs(predictions - actuals))
    mse = np.mean((predictions - actuals) ** 2)
    percentage_error = np.mean(np.abs(predictions - actuals) / actuals * 100)

    return mae, mse, percentage_error, z_score_error


def task_2(device):
    print("\n------------------ Task 2 -------------------")
    survival_dataset = SurvivalDataset("week-6/titanic.csv")

    train_size = int(0.8 * len(survival_dataset))
    test_size = len(survival_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        survival_dataset, [train_size, test_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = SurvivalPredictor().to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Start training model...")
    train_survival_model(model, train_loader, criterion, optimizer, device)

    print("\nEvaluating model performance...")
    train_accuracy = evaluate_survival_model(model, train_loader, device)
    test_accuracy = evaluate_survival_model(model, test_loader, device)

    print(f"Average Training Accuracy (survival): {train_accuracy:.2f}%")
    print(f"Average Test Accuracy (survival): {test_accuracy:.2f}%")
    print("-" * 45)


class SurvivalDataset(Dataset):
    def __init__(self, csv_file):
        dataframe = pd.read_csv(csv_file)
        dataframe["Sex"] = (dataframe["Sex"] == "male").astype(float)

        feature_columns = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]
        input_features = dataframe[feature_columns].fillna(
            dataframe[feature_columns].mean()
        )
        target_values = dataframe["Survived"].values

        self.scaler = StandardScaler()
        self.input_features = self.scaler.fit_transform(input_features)

        self.input_features = torch.FloatTensor(self.input_features)
        self.target_values = torch.FloatTensor(target_values)

    def __len__(self):
        return len(self.input_features)

    def __getitem__(self, idx):
        return self.input_features[idx], self.target_values[idx]


class SurvivalPredictor(nn.Module):
    def __init__(self, input_size=6, hidden_size=32):
        super(SurvivalPredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


def train_survival_model(model, train_loader, criterion, optimizer, device, epochs=100):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

        if (epoch + 1) % 10 == 0:
            accuracy = 100 * correct / total
            print(
                f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
            )


def evaluate_survival_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X).squeeze()
            predicted = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    return 100 * correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    task_1(device)
    task_2(device)


if __name__ == "__main__":
    main()
