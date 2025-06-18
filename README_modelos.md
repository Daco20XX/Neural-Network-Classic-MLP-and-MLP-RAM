
# 🤖 Classificação de Risco de Colisão com Redes Neurais (MLP & MLP-RAM)

Este repositório apresenta a etapa de modelagem e avaliação de duas arquiteturas de redes neurais aplicadas à tarefa de classificação de risco de colisão com base em dados extraídos de sensores.

---

## 🔍 Objetivo

Comparar o desempenho de uma rede neural perceptron multicamada tradicional (**MLP**) com uma arquitetura alternativa baseada em memória associativa (**MLP-RAM**), utilizando como base as características extraídas do projeto de pré-processamento.

---

## 📌 Funcionalidades

- Implementação de modelos MLP e MLP-RAM
- Treinamento e avaliação nos mesmos conjuntos de dados
- Cálculo de métricas de classificação
- Análise comparativa dos resultados

---

## 🛠 Tecnologias Utilizadas

- **Python 3.x**  
- **NumPy**  
- **Pandas**  
- **Scikit-learn**  
- **Matplotlib**  
- **PyTorch** (opcional, dependendo da implementação MLP-RAM)

---

## ▶️ Como Executar

1. Clone o repositório:
```bash
git clone https://github.com/SEU_USUARIO/NOME_DO_REPOSITORIO_MODELAGEM.git
cd NOME_DO_REPOSITORIO_MODELAGEM
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Execute os scripts dos modelos:
```bash
python mlp_tradicional.py
python mlp_ram.py
```

---

## 📁 Estrutura do Projeto

```
📦 classificacao-mlp-mlp-ram
├── 📜 mlp_tradicional.py
├── 📜 mlp_ram.py
├── 📁 dados
│   └── features_processadas.csv
├── 📁 resultados
│   └── metricas_mlp.txt
│   └── metricas_mlp_ram.txt
└── 📄 README.md
```

---

## 🧠 Estrutura dos Modelos

### MLP Tradicional

- Camada de entrada: 5 neurônios (correspondentes às features extraídas)
- 1 camada oculta com ativação ReLU
- Camada de saída com 2 neurônios (softmax)
- Otimizador: Adam
- Critério: CrossEntropyLoss

### MLP-RAM

- Utiliza unidades de memória RAM com endereçamento binário
- Cada entrada é convertida para formato binário com quantização
- Arquitetura não treinada por gradiente, mas por atualização direta das memórias

---

## 📈 Resultados Obtidos

### 🔸 MLP Tradicional

- **Acurácia**: 0.9800

```
              precision    recall  f1-score   support

   Sem Risco       0.98      1.00      0.99        81
   Com Risco       1.00      0.89      0.94        19

    accuracy                           0.98       100
   macro avg       0.99      0.95      0.97       100
weighted avg       0.98      0.98      0.98       100
```

**Matriz de Confusão:**
```
[[81  0]
 [ 2 17]]
```

---

### 🔹 MLP-RAM

- **Acurácia**: 0.8100

```
              precision    recall  f1-score   support

   Sem Risco       0.81      1.00      0.90        81
   Com Risco       0.00      0.00      0.00        19

    accuracy                           0.81       100
   macro avg       0.41      0.50      0.45       100
weighted avg       0.66      0.81      0.72       100
```

**Matriz de Confusão:**
```
[[81  0]
 [19  0]]
```

---

## 📊 Análise Crítica

- A **MLP tradicional** teve desempenho excelente, com alta precisão e recall para ambas as classes, sendo capaz de identificar corretamente a maioria das situações de risco.
- A **MLP-RAM**, por outro lado, apresentou um comportamento enviesado: classificou todas as amostras como "Sem risco", o que indica um problema de generalização para a classe minoritária.
- Possíveis causas para o desempenho inferior da MLP-RAM:
  - Falta de capacidade de representar relações complexas entre as features
  - Sensibilidade à quantização binária
  - Necessidade de ajuste nos parâmetros de endereçamento e número de unidades RAM

---

## 📎 Conclusão

Este experimento mostrou que a abordagem MLP tradicional é mais adequada para este conjunto de dados. A MLP-RAM, apesar de interessante por sua simplicidade e velocidade, requer melhorias para ser viável nesta aplicação.

