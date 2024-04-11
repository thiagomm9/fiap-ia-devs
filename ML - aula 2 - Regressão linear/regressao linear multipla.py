# IMPORTANDO OS DADOS
import pandas as pd

pd.set_option('display.max_columns', None)  # Para mostrar todas as colunas
dataset = pd.read_csv('housing.csv')
print(dataset.head(), '\n')

# ANALISANDO OS DADOS
# para deixar todas as saídas com os mesmos valores obtidos na live.
import numpy as np

np.random.seed(42)  # instanciando uma semente aleatória para manter a constância da base
import os

# Para plots bonitinhos
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

print(dataset.shape)  # 20640 linha por 10 colunas
dataset.info()
# Opa! Observe aqui que total_bedrooms possui 20433 linhas de informações preenchidas (não nulos).
# Isso significa que 207 bairros não possuem características.

# A única variável do tipo texto é "ocean_proximity". Vamos analisar quantas categorias existem e quantos bairros
# pertencem a essas categorias utilizando a função value_counts()?
print('\n')
print(set(dataset['ocean_proximity']))
print(dataset['ocean_proximity'].value_counts(), '\n')

# Agora vamos analisar os dados do tipo numérico com a função describe():
print(dataset.describe())

# Analisando algumas distribuições com histogramas:
# Atentar aqui para a uniformidade dos dados e verificar outliers
# %matplotlib inline
import matplotlib.pyplot as plt
dataset.hist(bins=50, figsize=(20, 15))
# plt.show()

### SEPARANDO AS BASES EM TREINO E TESTE
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(dataset, test_size=0.2, random_state=7)
print('\n', len(df_train), 'treinamento +', len(df_test), 'teste\n')

### CRIANDO CATEGORIAS DE MÉDIA DE RENDA ANUAL
# Pode-se criar categorias para qualquer valor contínuo,
# para que se distribua os valores igualitariamente entre as bases
# Vamos supor que algum especialista da área de vendas de imóveis notificou que a média de renda anual
# é um atributo importante para colocar no modelo preditivo para estimar preços médios.
# Quando dividimos o conjunto de treino e teste precisamos garantir que ambos sejam
# representativos com todos os valores de renda anual.Como a média de renda anual é um atributo numérico,
# que tal criar uma categoria de renda ?
dataset["median_income"].hist()
# plt.show()
# Divida por 1,5 para limitar o número de categorias de renda
# dividindo o valor da coluna "median_income" de cada entrada pelo valor 1,5 e,
# em seguida, arredondando o resultado para cima usando a função
# np.ceil() (da biblioteca NumPy). Isso cria uma nova coluna chamada "income_cat"
# no dataset que contém os valores das categorias de renda após a divisão e arredondamento.
dataset["income_cat"] = np.ceil(dataset["median_income"] / 1.5)
# Rotule aqueles acima de 5 como 5.
# os valores na coluna "income_cat" que forem maiores ou iguais a 5
# são substituídos por 5. Isso é feito usando a função .where() do pandas.
# Basicamente, se o valor em "income_cat" for menor que 5, ele permanece o mesmo; caso contrário, é substituído por 5.
dataset["income_cat"].where(dataset["income_cat"] < 5, 5.0, inplace=True)
# cut do Pandas, que é comumente usada para dividir um conjunto de dados em intervalos
# discretos chamados de "bins" (intervalos ou faixas)
dataset["income_cat"] = pd.cut(dataset["median_income"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])
# bins(valor) distribui para labels: 0. à 1,4 = cat 1, 1.5 à 2.9 cat 2...6. ou mais = cat 5
print(dataset["income_cat"].value_counts())
dataset["income_cat"].hist()
# plt.show()
# Resumindo, esse código está transformando valores contínuos de renda em categorias discretas,
# dividindo-os em intervalos específicos e arredondando-os para cima, garantindo que
# o número de categorias seja limitado e, finalmente, atribuindo rótulos numéricos a essas categorias.
# Boa! Agora com as categorias criadas, vamos realizar a amostragem estratificada com base na categoria de renda!
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(dataset, dataset["income_cat"]):
    strat_train_set = dataset.loc[train_index]
    strat_test_set = dataset.loc[test_index]
# Analisando as proporções (verificando se as proporções estão semelhantes em todas as bases)
print('\n')
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
# Analisando as proporções
print(strat_train_set["income_cat"].value_counts() / len(strat_train_set))
print(dataset["income_cat"].value_counts() / len(dataset))
# Show! Depois de garantir que os valores médios de renda anual estão distribuídos de forma estratificada,
# podemos remover a coluna income_cat que utilizamos como variável auxiliar.
# Removendo o income_cat das bases de treino e teste
# O uso do termo set_ é uma convenção para indicar que é uma variável temporária
# que itera sobre um conjunto de dados (um conjunto de treinamento ou um conjunto de teste).
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

# ANALISANDO DADOS GEOGRÁFICOS
housing = strat_train_set.copy()  # Realiza uma cópia da base
housing.plot(kind="scatter", x="longitude", y="latitude")
# plt.show()
# Com alpha 0.1 setado, é possível analisar a concentração de dados
# em determinados pontos do gráfico (densos e não densos)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
# plt.show()
# A concentração das localidades no gráfico terá um formato semelhante ao do estado da Califórnia,
# que é a localidade dos dados

# ANALISANDO PREÇOS IMOBILIÁRIOS
# s = a concentração da população em cada região
# c = valor do imovel
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
# plt.show()

# BUSCANDO CORRELAÇÕES
corr_matrix = housing.corr(numeric_only=True)
print(
    corr_matrix["median_house_value"].sort_values(ascending=False)
)  # Compara a correlação das variáveis declarativas com a target
from pandas.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
# Une histogramas com gráficos de dispersão
# as variáveis com maior correlação terão um crescimento parecido
# plt.show()
# Analisando as correlações, a feature que seja mais promissora para prever o valor médio da habitação é a renda média.
# median_income : 0.68 de correlação.
# Vamos plotar essas duas features em um gráfico de scatter para analisar com mais detalhes:
housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0, 16, 0, 550000])
plt.show()
# Observamos que:
# A correlação realmente mostra uma certa tendência ascendente nos dados e os pontos não estão mais dispersos.
# O limite de preço que temos na base de dados é claramente visível como uma linha horizontal em 500 mil dólares.
# Observe também que temos essas linhas retas (claro que menos óbvias) na horizontal em torno de 450 mil dólares,
# outra em 350 mil dólares e uma em 280 mil dólares.
# Será que se deixarmos essas peculiaridades nos dados,
# não pode afetar nosso algoritmo?

### PREPARANDO OS DADOS PARA COLOCAR NO ALGORITMO
