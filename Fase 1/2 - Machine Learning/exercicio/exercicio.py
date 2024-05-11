"""Chegou a hora de compilar todos os aprendizados obtidos nas aulas de Machine Learning, em um único lugar,
para fazer a entrega do Desafio da disciplina. Utilizando a base de dados “insurance.csv”, você tem o desafio de
criar um modelo preditivo de regressão para prever o valor dos custos médicos individuais cobrados pelo seguro de saúde.

Sobre a base de dados Essa base de dados contém 1.338 linhas com informações sobre a idade da pessoa, gênero,
índice de massa corporal (IMC), número de filhos, flag de verificação se a pessoa é fumante, região residencial do
benefício e o valor do custo médico.

Dicionário dos dados
Idade: idade do beneficiário principal.

Gênero: gênero do contratante de seguros.

IMC: índice de massa corporal, fornecendo uma compreensão do corpo, pesos relativamente altos ou baixos em relação à
altura.

Filhos: número de filhos cobertos por seguro saúde / Número de dependentes.

Fumante: se a pessoa fuma (sim ou não).

Região: a área residencial do beneficiário nos EUA, nordeste, sudeste, sudoeste ou noroeste.

Encargos: custos médicos individuais cobrados pelo seguro de saúde.

Objetivo
Criar um modelo preditivo e comprovar sua eficácia com métricas estatísticas.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Read insurance.csv file to data frame
df = pd.read_csv("insurance.csv")
# print(df.shape)
# print(df.info())

# Check for missing values
# print(df.isnull().sum())

# Get the different values the column region
# print(df['region'].unique())

# Get the different values the column sex
# print(df['sex'].unique())

# Get the different values the column smoker
# print(df['smoker'].unique())

# Visualiza histograma
df.hist(bins=50, figsize=(20, 15))
plt.show()

# label encode necessary columns
le = LabelEncoder()
df['region'] = le.fit_transform(df['region'])
df['sex'] = le.fit_transform(df['sex'])
df['smoker'] = le.fit_transform(df['smoker'])

# Convert categorical variables to numerical using one-hot encoding
# df = pd.get_dummies(df)

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot correlation matrix
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.6)
# plt.show()

# Split db to X and y
X = df.drop(columns=['charges'])  # Features
y = df['charges']  # Target

# Split db to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model with linear regression
model = LinearRegression()
model.fit(X_train, y_train)

# Calculate MSE
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mse = np.sqrt(mse)  # raiz quadrada aqui
print(mse)

# Calculate MAE
mae = mean_absolute_error(y_test, y_pred)
print(mae)

# Calculate r2 score
r2 = r2_score(y_test, y_pred)
print('r²', r2)


# Função para calcular o MAPE (Mean Absolute Percentage Error)
def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape


# Calcular o MAPE
mape_result = calculate_mape(y_test, y_pred)

# Imprimir o resultado
print(f"O MAPE é: {mape_result:.2f}%")
