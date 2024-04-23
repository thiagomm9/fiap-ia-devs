"""
Exercício prático
Dataset sobre atributos dos jogadores no jogo eletrônico de esporte FIFA 2022
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro

pd.set_option('display.max_columns', None)

# Vamos carregar o arquivo csv em nosso drive e analisarmos o nosso dataframe
df_fifa = pd.read_csv("players_22.csv")

# Imprime as primeiras duas linhas
# print(df_fifa.head(2))

# Tamanho da base de dados
# print(df_fifa.shape)

# Analisando informações básicas sobre nosso dataframe como tipos de variáveis, tamanho,
# amostra estatísticas básicas e etc...
# print(df_fifa.describe())

print("Informações sobre o DataFrame df_fifa:")
print(df_fifa.info())

"""Temos muitas variáveis como analisamos, suponha que preciso treinar um modelo de clustering com este conjunto de 
dados, é crucial reduzir a dimensionalidade. Isso ajuda o modelo a focar nos aspectos mais relevantes, simplificando 
a interpretação e destacando padrões significativos. Vamos então analisar se realmente podemos aplicar o PCA em 
nossos dados."""

"""Matriz de correlação 
A matriz de correlação é como um mapa que nos mostra o quão próximo ou distante diferentes 
variáveis estão umas das outras em um conjunto de dados. Vamos gerar uma com nossos dados para entender melhor seu 
funcionamento."""

# Criando dataframe somente com nossas variáveis numericas / A matriz de correlacao so funciona com dados numericos
df_fifa_numerico = df_fifa.select_dtypes([np.number])
# Calcula a matriz de correlação
correlation_matrix = df_fifa_numerico.corr()

# Visualização da matriz de correlação
plt.figure(figsize=(10, 8))
# Criando grafico de mapa de calor com seaborn
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f", linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()

"""Em cada célula dessa tabela, temos um número que varia de -1 a 1, representando o quão forte é a relação entre 
duas variáveis. Um número próximo de 1 sugere uma forte conexão positiva, enquanto um número próximo de -1 indica uma 
forte conexão negativa. Aquela linha diagonal que sempre contém o número 1? Bem, ela mostra que uma variável é 
perfeitamente relacionada consigo mesma. 
Ao examinarmos a matriz de correlação, analisamos as interações entre as 
variáveis de forma mais detalhada. Essa análise é crucial ao considerar a aplicação da Análise de Componentes 
Principais (PCA), que procura identificar relações lineares entre variáveis. A matriz de correlação atua como uma 
ferramenta que evidencia essas conexões. Quando as variáveis estão alinhadas de maneira coesa, o PCA desempenha um 
papel semelhante ao de um guia turístico, simplificando a exploração do conjunto de dados. No entanto, 
se as variáveis seguem caminhos independentes, outras estratégias de redução de dimensionalidade podem ser mais 
apropriadas."""

# A quantidade de dimensoes/variaveis descritivas inviabiliza uma analise precisa dos dados

# Vamos verificar quantidades de nulos em nosso dataframe, não podemos aplicar o PCA
# se nossos dados tiverem linhas nulas, devemos tratar esses casos caso ocorram.
# Existem diversas técnicas para tratar dados nulos, o método de escolha depende muito do seu objetivo
print(df_fifa_numerico.isnull().sum())

# Preenche os valores NaN com a média das colunas
imputer = SimpleImputer(strategy='mean')  # media
df_fifa_numerico = pd.DataFrame(imputer.fit_transform(df_fifa_numerico), columns=df_fifa_numerico.columns)

# Padroniza as variáveis
scaler = StandardScaler()
df_fifa_padronizado = scaler.fit_transform(df_fifa_numerico)
# Calcula a variância explicada acumulada
pca = PCA()
pca.fit(df_fifa_padronizado)
variancia_cumulativa = np.cumsum(pca.explained_variance_ratio_)
# Visualização da variância explicada acumulada
plt.plot(range(1, len(variancia_cumulativa) + 1), variancia_cumulativa, marker='o')
plt.xlabel('Número de Componentes Principais')
plt.ylabel('Variância Acumulada Explicada')
plt.title('Variância Acumulada Explicada pelo PCA')
plt.show()

# Vamos definir um limiar de 80%, ou seja, queremos obter uma porcentagem de explicancia sobre
# nossos dados de igual a 80%
limiar_de_variancia = 0.80

# Encontrar o número de componentes necessários para atingir ou ultrapassar o limiar
num_de_pca = np.argmax(variancia_cumulativa >= limiar_de_variancia) + 1

print(f"Número de Componentes para {limiar_de_variancia * 100}% da Variância: {num_de_pca}")
# Número de Componentes para 80.0% da Variância: 10

# Por fim vamos então utilizar nosso número de PCA desejado e reduzir nossas 59 columns para 10
# Inicializa o objeto PCA
pca = PCA(n_components=num_de_pca)
# Aplica o PCA aos dados padronizados
principal_components = pca.fit_transform(df_fifa_padronizado)

# Exibe a proporção de variância explicada
explained_variance_ratio = pca.explained_variance_ratio_
print(explained_variance_ratio)

# Pegando o número de componentes principais gerados
num_components = principal_components.shape[1]
# Gerando uma lista para cada PCA
column_names = [f'PC{i}' for i in range(1, num_components + 1)]
# Criando um novo dataframe para visualizarmos como ficou nossos dados reduzidos com o PCA
pca_df = pd.DataFrame(data=principal_components, columns=column_names)

# VERIFICANDO NORMALIDADE APOS O PCA
# Criar histogramas para cada coluna
plt.figure(figsize=(15, 8))
for i, col in enumerate(pca_df.columns[:10]):
    plt.subplot(2, 5, i + 1)  # Aqui, ajustei para 2 linhas e 5 colunas
    sns.histplot(pca_df[col], bins=20, kde=True)
    plt.title(f'Histograma {col}')
plt.tight_layout()
plt.show()

"""Se as distribuições das suas variáveis (PC1 a P10) não seguem uma curva gaussiana (distribuição normal), 
isso pode impactar a interpretação de algumas análises estatísticas que pressupõem normalidade. Entretanto, 
nem sempre é necessário que os dados sigam uma distribuição normal, especialmente se você estiver utilizando métodos 
não paramétricos ou técnicas robustas que não dependem dessa suposição."""


# Vamos olhar para cada coluna a normalidade após a redução de dimensionalidade
for column in pca_df.columns:
    stat, p_value = shapiro(pca_df[column])
    print(f'Variável: {column}, Estatística de teste: {stat}, Valor p: {p_value}')
    # Você pode então interpretar o valor p para determinar se a variável segue uma distribuição normal
    if p_value > 0.05:
        print(f'A variável {column} parece seguir uma distribuição normal.\n')
    else:
        print(f'A variável {column} não parece seguir uma distribuição normal.\n')

"""
PROXIMOS PASSOS
Muitos algoritmos de clustering, como o K-Means, assumem que os dados seguem uma distribuição esférica e têm variação 
constante em todas as direções. Se as variáveis que você está usando para clustering não seguem uma distribuição 
normal, isso pode afetar a performance do algoritmo.
No entanto, vale ressaltar que a sensibilidade dos algoritmos de cluster à normalidade dos dados pode variar. Alguns 
algoritmos, como o K-Means, podem ser sensíveis à escala e formato dos clusters. Outros algoritmos, como o DBSCAN (
Density-Based Spatial Clustering of Applications with Noise), são mais flexíveis em relação à forma e densidade dos 
clusters.
"""
