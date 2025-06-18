import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, ClassifierMixin


# 🔧 Classe MLP-RAM simples
class MLPRAM(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, n_rams=8, bits_per_ram=4):
        self.threshold = threshold
        self.n_rams = n_rams
        self.bits_per_ram = bits_per_ram
        self.rams = [{} for _ in range(n_rams)]

    def binarize(self, x):
        # Normaliza para [0, 1], converte para binário
        return np.round(x).astype(int)

    def _split_input(self, binary_input):
        # Divide o vetor binário em pedaços (endereços de RAM)
        return np.array_split(binary_input, self.n_rams)

    def fit(self, X, y):
        X_bin = self.binarize(X * (2 ** (self.n_rams * self.bits_per_ram) - 1)).astype(int)
        for xi, target in zip(X_bin, y):
            chunks = self._split_input(np.unpackbits(xi.view(np.uint8)))
            for i, chunk in enumerate(chunks):
                addr = tuple(chunk)
                if addr not in self.rams[i]:
                    self.rams[i][addr] = [0, 0]
                self.rams[i][addr][target] += 1
        return self

    def predict(self, X):
        X_bin = self.binarize(X * (2 ** (self.n_rams * self.bits_per_ram) - 1)).astype(int)
        preds = []
        for xi in X_bin:
            chunks = self._split_input(np.unpackbits(xi.view(np.uint8)))
            votes = [0, 0]
            for i, chunk in enumerate(chunks):
                addr = tuple(chunk)
                if addr in self.rams[i]:
                    for j in [0, 1]:
                        votes[j] += self.rams[i][addr][j]
            pred = 1 if votes[1] >= votes[0] else 0
            preds.append(pred)
        return np.array(preds)

# 1. Carrega os dados
df = pd.read_csv("distancia_colisao.csv")
X = df[["distancia_cm"]].values
y = df["risco_colisao"].values

# 2. Normaliza e divide
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 3. Treina MLP-RAM
print("Treinando MLP-RAM...")
ram_model = MLPRAM(n_rams=4, bits_per_ram=2)
ram_model.fit(X_train, y_train)

# 4. Avalia
y_pred = ram_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🔹 Acurácia MLP-RAM: {acc:.4f}")
print("\n📄 Relatório de Classificação:")
print(classification_report(y_test, y_pred))
print("\n📌 Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

