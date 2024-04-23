# ESCALONAMENTO DE DADOS
# Afinal, porque as escalas dos dados são importantes?


"""Para definirmos o que seriam as escalas dos dados, podemos dizer que as escalas referem-se à amplitude ou
intervalo dos valores de um conjunto numérico de dados.

A técnica "feature scaling" é o processo de normalizar as escalas das features, colocando-as em uma escala comum.
Existem duas abordagens comuns para fazer isso: a normalização (também conhecida como min-max scaling) e a
padronização (também conhecida como z-score normalization). No aprendizado de máquina, muitos algoritmos podem ser
sensíveis às escalas dos dados, podendo se confundir e achar que as escalas maiores, por exemplo, são mais relevantes
do que as variáveis de escalas menores. Esse problema pode afetar principalmente algoritmos de redes neurais,
onde a normalização dessas variáveis é obrigatória para a representação da dimensão ficar em uma mesma escala e
contribuir com a convergência mais rápida dos dados. Não somente em deep learning essa técnica pode ajudar,
mas também com algoritmos que lidam com distâncias, como o k-means, KNN, PCA, SVM e regressão logística por exemplo.

Este conjunto de dados que vamos utilizar na aula de hoje, contém detalhes dos clientes de um banco e a variável alvo
é uma variável binária que reflete o fato de o cliente ter deixado o banco (fechado sua conta) ou continuar a ser
cliente.

É o famoso modelo de churn!

Nessa aula vamos criar um modelo preditivo para prever se o cliente vai deixar o banco, e é claro, que vamos testar
nossos dados com e sem escalonamento.

Bem, logo já percebemos aqui que essa base de dados nos traz um desafio para ser resolvido com modelos
supervisionados. Nessa aula não iremos focar muito nos detalhes dos modelos, mas não preocupe que na aula de Machine
Learning Avançado você irá conhecer cada detalhe!

Vamos nessa aula focar em padronização e normalização, utilizando essa base como exemplo.

Vamos realizar o upload dessa base de dados para começar a construir o modelo!"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Churn_Modelling.csv", sep=";")
print(df.head())
print(df.shape)

"""Observe que essa base de dados contém um grande número de features (ao total são 14). Dentre essas features 
disponíveis na base, muitas possuem proporções diferentes quando analisamos as escalas dos dados, por exemplo, idade, 
salário, o número de posse da pessoa, número de produtos, o score de crédito e assim por diante. Será que essas 
variáveis têm uma amplitude tão diferente, pode impactar nosso modelo preditivo? Bem, como próximo passo, 
vamos plotar alguns boxplots para analisarmos como está a distribuição dessas variáveis quantitativas para 
descobrirmos a variação de amplitude dos dados:"""

# Criar o gráfico de boxplot (os valores(bolinhas) fora do min e max sao outliers)
# O traco colorido no meio do quadro corresponde a mediana
plt.boxplot(df['CreditScore'])
plt.title('CreditScore')
plt.ylabel('Valores')
plt.show()

print(df['CreditScore'].min())
print(df['CreditScore'].max())

# Criar o gráfico de boxplot
# Neste casa ha muitos outliers acima do limite superior
plt.boxplot(df['Age'])
plt.title('Age')
plt.ylabel('Valores')
plt.show()

print(df['Age'].min())
print(df['Age'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['Tenure'])
plt.title('Tenure')
plt.ylabel('Valores')
plt.show()

print(df['Tenure'].min())
print(df['Tenure'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['Balance'])
plt.title('Balance')
plt.ylabel('Valores')
plt.show()

print(df['Balance'].min())
print(df['Balance'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['NumOfProducts'])
plt.title('NumOfProducts')
plt.ylabel('Valores')
plt.show()

print(df['NumOfProducts'].min())
print(df['NumOfProducts'].max())

# Criar o gráfico de boxplot
plt.boxplot(df['EstimatedSalary'])
plt.title('EstimatedSalary')
plt.ylabel('Valores')
plt.show()

print(df['EstimatedSalary'].min())
print(df['EstimatedSalary'].max())

"""OK, concluímos que as escalas são bem diferentes! Falando sobre transformações nos dados, os dados em formato de 
string (as categorias) também passam por um tipo de transformação. Vamos aplicar nos dados em formato de texto o 
LabelEncoder. O LabelEncoder transforma rótulos de classes em números inteiros. Mas por que é importante fazer esse 
tipo de transformação nas categorias? Para os algoritmos de machine learning funcionarem, é necessário transformar a 
informação em um formato numérico para que o computador possa compreender o que estamos querendo apresentar."""

print(df.head())
label_encoder = LabelEncoder()
# Ajustar e transformar os rótulos (Transforma as variaveis texto em numericas)
df['Surname'] = label_encoder.fit_transform(df['Surname'])
df['Geography'] = label_encoder.fit_transform(df['Geography'])
df['Gender'] = label_encoder.fit_transform(df['Gender'])
print(df.head())

# OK, como próximo passo, antes de normalizar ou padronizar, vamos separar os dados em treino e teste:
X = df.drop(columns=['Exited'])  # Variáveis características
y = df['Exited']  # O que eu quero prever. (Target)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""Quanto a esses detalhes de separação de dados em treino e teste, não se preocupe que em Machine Learning avançado 
você irá aprender cada passo! Vamos escalonar os dados? Importamos os escalonadores da biblioteca sklearn em 
preprocessing o StandardScaler e MinMaxScaler:"""
# APLICANDO O MINMAX SCALER

"""O escalonamento min-max (normalização) é muito simples! Basicamente os valores são deslocados e redimensionados 
para que acabem variando de 0 a 1. Esse cálculo subtrai o valor mínimo e divide pelo 
máximo, menos o mínimo."""

scaler = MinMaxScaler()  # chamando o metodo de normalização dos dados (0-1)

scaler.fit(X_train)

x_train_min_max_scaled = scaler.transform(X_train)
x_test_min_max_scaled = scaler.transform(X_test)

"""Mas você deve estar se perguntando:

“por que é realizado o escalonamento (fit) na base treino e não na base de teste?”.

Bem, realizamos a transformação do escalonamento na base de treino para evitar que a base de teste fique exatamente 
igual às estatísticas da base de treino, o que evita “vazamento” desses dados. A base de teste em geral deve 
representar uma base de dados nunca vista antes pelo algoritmo, justamente para testar se o algoritmo consegue 
generalizar os dados."""

# AGORA VAMOS TESTAR O STANDARD SCALER

"""A padronização não vincula valores específicos nos mínimos e máximos, o que pode ser um ponto de atenção em alguns 
algoritmos (por exemplo, redes neurais). No entanto, a padronização é muito menos afetada por outliers. Na biblioteca 
do Scikit-Learn temos a padronização em StandardScaler."""

scaler = StandardScaler()  # chamando o metodo de padronização dos dados (média e std)

scaler.fit(X_train)  # qual média e std será utilizado para o escalonamento

x_train_standard_scaled = scaler.transform(X_train)
x_test_standard_scaled = scaler.transform(X_test)

# Vamos testar o algoritmo sem os escalonadores e validar os resultados!

model = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar a precisão do modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Acurácia: {accuracy:.2f}')

# Testando com a normalização:
model_min_max = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_min_max.fit(x_train_min_max_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_min_max = model.predict(x_test_min_max_scaled)

accuracy_min_max = accuracy_score(y_test, y_pred_min_max)
print(f'Acurácia: {accuracy_min_max:.2f}')

# Testando com a padronização:
model_standard = KNeighborsClassifier(n_neighbors=3)

# Treinar o modelo
model_standard.fit(x_train_standard_scaled, y_train)

# Fazer previsões no conjunto de teste
y_pred_standard = model.predict(x_test_standard_scaled)

accuracy_strandard = accuracy_score(y_test, y_pred_standard)
print(f'Acurácia: {accuracy_strandard:.2f}')
