import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class DeepLearningChatbot(nn.Module):

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.net(x)

    def train_model(self, X_train, y_train, epochs=10, batch_size=32, lr=1e-3):

        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(1, epochs+1):
            total_loss = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = self(X_batch)
                loss = loss_fn(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/{epochs} - Loss: {total_loss/len(loader):.4f}")

    def predict(self, X):

        self.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            probs = self(X_tensor)
            return torch.argmax(probs, dim=-1).tolist()


def test_chatbot2():

    # Exemples de features : [cote_match, forme_equipe1, forme_equipe2]
    X_train = [
        [1.5, 0.8, 0.3],
        [2.2, 0.4, 0.6],
        [1.8, 0.7, 0.5],
        [3.0, 0.2, 0.9]
    ]
    # Labels : 0 (éviter pari), 1 (pari modéré), 2 (pari recommandé)
    y_train = [2, 0, 1, 0]

    bot2 = DeepLearningChatbot(input_dim=3, hidden_dim=16, output_dim=3)
    bot2.train_model(X_train, y_train, epochs=5, batch_size=2)

    # Test de prédiction
    X_test = [
        [2.0, 0.6, 0.4],
        [1.4, 0.9, 0.2]
    ]
    preds = bot2.predict(X_test)
    for features, p in zip(X_test, preds):
        print(f"Features {features} -> Classe prédite {p}")

if __name__ == "__main__":
    test_chatbot2()
