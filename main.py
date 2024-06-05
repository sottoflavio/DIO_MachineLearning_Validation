import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

iris = load_iris()
X = iris.data
y = iris.target

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o classificador
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Calcular as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Sensibilidade e Especificidade por classe
sensibilidade = recall_score(y_test, y_pred, average=None)
especificidade = []
for i in range(len(cm)):
    tn = sum(sum(cm)) - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
    fp = cm[:, i].sum() - cm[i, i]
    especificidade.append(tn / (tn + fp))

print(f"Matriz de Confusão:\n{cm}")
print(f"Acurácia: {accuracy:.2f}")
print(f"Precisão: {precision:.2f}")
print(f"Recall (Sensibilidade): {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"Sensibilidade por classe: {sensibilidade}")
print(f"Especificidade por classe: {especificidade}")
