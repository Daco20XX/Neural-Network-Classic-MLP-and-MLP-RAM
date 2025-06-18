import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Carregar os dados
print("🔹 Carregando o dataset...")
df = pd.read_csv("distancia_colisao.csv")

# 2. Preparar entrada e saída
print("🔹 Separando variáveis de entrada e saída...")
X = df[["distancia_cm"]].values
y = df["risco_colisao"].values

# 3. Normalizar os dados
print("🔹 Normalizando a coluna de distância (0 a 1)...")
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Dividir em treino e teste
print("🔹 Dividindo os dados em treino e teste (80% treino / 20% teste)...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Criar e treinar o modelo MLP
print("🔹 Treinando o modelo MLPClassifier (scikit-learn)...")
mlp = MLPClassifier(hidden_layer_sizes=(8, 4), activation='relu', solver='adam', max_iter=500, random_state=42)
mlp.fit(X_train, y_train)

# 6. Avaliar desempenho do modelo
print("\n✅ Modelo treinado com sucesso!\n")
print("📊 Avaliando desempenho nos dados de teste...")
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n🔸 Acurácia do modelo: {acc:.4f}")

print("\n📄 Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=["Sem Risco", "Com Risco"]))

print("📌 Matriz de Confusão:")
conf_matrix = confusion_matrix(y_test, y_pred)
print(conf_matrix)

# 7. Visualizar decisão do modelo com gráfico
print("\n📈 Gerando gráfico de decisão do modelo...")
dist_range = np.linspace(2, 400, 1000).reshape(-1, 1)
dist_range_scaled = scaler.transform(dist_range)
preds = mlp.predict_proba(dist_range_scaled)[:, 1]

plt.figure(figsize=(8, 4))
plt.plot(dist_range, preds, label="Probabilidade de Risco", color='blue')
plt.axvline(100, color='red', linestyle='--', label='Limite de risco (100 cm)')
plt.xlabel("Distância (cm)")
plt.ylabel("Probabilidade de risco de colisão")
plt.title("Decisão do MLP sobre o risco de colisão")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
