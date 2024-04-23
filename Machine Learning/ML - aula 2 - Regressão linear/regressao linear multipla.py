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
plt.show()

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
plt.show()
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
plt.show()
# Com alpha 0.1 setado, é possível analisar a concentração de dados
# em determinados pontos do gráfico (densos e não densos)
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
plt.show()
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
plt.show()

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
housing = strat_train_set.drop("median_house_value", axis=1)  # apagando a target para a base de treino (nosso x)
housing_labels = strat_train_set["median_house_value"].copy()  # armazenando a target (nosso y)
# listando as colunas nulas
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
print(sample_incomplete_rows)
print(housing.isnull().sum())
# OK, como vamos tratar esses valores nulos?
# Opção 1
# Substituindo os valores nulos pela mediana
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True)
print(sample_incomplete_rows)

# UTILIZANDO AS CLASSES DO SKLEARN
# Você também pode optar por utilizar classes acessíveis do Sklearn!
# Você pode criar pipelines de pré-processamento e modelagem com facilidade usando as classes do Scikit-Learn.
# Isso permite criar fluxos de trabalho mais organizados e repetíveis.
# Vamos utilizar o Imputer para substituir os valores faltantes pela média.
# Opção 2
try:
    from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+
except ImportError:
    from sklearn.preprocessing import Imputer as SimpleImputer
imputer = SimpleImputer(strategy="median")

# Remova o atributo de texto porque a mediana só pode ser calculada em atributos numéricos:
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)  # calculando a mediana de cada atributo e armazenando o resultado na variável statistics_

print('\n')
print(imputer.statistics_)

# Verifique se isso é o mesmo que calcular manualmente a mediana de cada atributo:
print(housing_num.median().values)

# Aplicando o Imputer "treinado" na base para substituir valores faltantes perdidos pela mediana:
X = imputer.transform(housing_num) # o resultado é um array.
print(X)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
print(housing_tr)

# verificando os resultados
print(housing_tr.loc[sample_incomplete_rows.index.values])

print(imputer.strategy)

housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
print(housing_tr.head())


# PRÉ-PROCESSANDO AS CATEGORIAS
# Agora vamos pré-processar o recurso de entrada categórica, ocean_proximity:
housing_cat = housing[['ocean_proximity']]
print(housing_cat.head(10))

# O OrdinalEncoder é uma classe da biblioteca scikit-learn, usada para transformar variáveis categóricas ordinais
# em valores numéricos. Variáveis ordinais são aquelas que têm uma ordem ou hierarquia específica,
# mas as distâncias entre os valores não são necessariamente significativas.
try:
    from sklearn.preprocessing import OrdinalEncoder
except ImportError:
    from future_encoders import OrdinalEncoder # Scikit-Learn < 0.20
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])

print(ordinal_encoder.categories_)

# O OneHotEncoder é outra classe da biblioteca scikit-learn,
# usada para transformar variáveis categóricas em representações numéricas binárias.
# Ele é particularmente útil quando se lida com variáveis categóricas nominais,
# ou seja, aquelas que não têm uma ordem específica.
try:
    from sklearn.preprocessing import OrdinalEncoder # apenas para gerar um ImportError se Scikit-Learn < 0.20
    from sklearn.preprocessing import OneHotEncoder
except ImportError:
    from future_encoders import OneHotEncoder # Scikit-Learn < 0.20

cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)
print(cat_encoder.categories_)

# CRIANDO A PIPELINA DE PRÉ-PROCESSAMENTO
# Agora vamos construir um pipeline para pré-processar os atributos numéricos:
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")), #substituindo valores nulos pela mediana
    ('std_scaler', StandardScaler()), # padronizando as escalas dos dados
])
housing_num_tr = num_pipeline.fit_transform(housing_num)
print(housing_num_tr)

# Agora, vamos tratar os valores categóricos:
# O ColumnTransformer é uma classe da biblioteca scikit-learn em Python
# que permite aplicar transformações específicas a diferentes colunas de um conjunto de dados
# (dados numéricos, categóricos, etc.) e deseja aplicar diferentes pré-processamentos
# ou transformações a cada tipo de coluna.
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer # Scikit-Learn < 0.20

from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),  # tratando as variáveis numéricas (chamando a pipeline de cima)
    ("cat", OneHotEncoder(), cat_attribs),  # tratando as variáveis categóricas
])
housing_prepared = full_pipeline.fit_transform(housing)

print(housing_prepared)

print('\n')
print(housing_prepared.shape)
print(type(housing_prepared))


# Perceba que o resultado é uma matriz multidimensional. Precisamos transformá-la em dataframe.
column_names = [
    'longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
    'population', 'households', 'median_income', '<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
# Transformar o array em DataFrame
housing_df = pd.DataFrame(data=housing_prepared, columns=column_names)
# Exibir o DataFrame resultante
print(housing_df.shape)

print(housing_df.head())
print(housing_df.isnull().sum())






# ESCOLHENDO O MELHOR MODELO DE REGRESSÃO
# Vamos começar com a velha e boa regressão linear!
# - Equação do 1° grau.
# - A Regressão Linear busca entender o padrão de um valor dependendo de outro ou outros, e assim encontrar uma função que expressa esse padrão.
# - Foco: buscar o melhor valor que os coeficientes possam atingir, de maneira que a diferença entre o valor predito pela função e o real, sejam os menores.
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)  # Treinamento

# vamos tentar o pipeline de pré-processamento completo em algumas instâncias de treinamento
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
predictions = lin_reg.predict(housing_prepared)
print("Predictions:", lin_reg.predict(some_data_prepared))

# Compare com os valores reais:
print("Labels:", list(some_labels))


# AVALIANDO O MODELO
# O MSE mede a média dos quadrados das diferenças entre os valores previstos pelo modelo
# e os valores reais observados no conjunto de dados.
# Quanto menor o valor do MSE, melhor o ajuste do modelo aos dados.
from sklearn.metrics import mean_squared_error
# erro médio quadrático eleva ao quadrado a média do erro médio absoluto.
# Estou avaliando se os erros não são tão grandes, esses erros são penalizados.
# penaliza muito mais valores distantes da média.
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)  # raiz quadrada aqui
print(lin_rmse)

from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(lin_mae)

# Um erro de margem de 69050 dólares não é muito aceitável no nosso modelo
# sendo que os valores de median_housing_values variam entre 120 mil dólares e 265 mil dólares
# Podemos definir aqui que esse modelo está com overfiting.
# Vamos tentar um modelo mais poderoso?
from sklearn.metrics import r2_score
r2 = r2_score(housing_labels, housing_predictions)
print('r²', r2)


# Função para calcular o MAPE (Mean Absolute Percentage Error)
def calculate_mape(labels, predictions):
    errors = np.abs(labels - predictions)
    relative_errors = errors / np.abs(labels)
    mape = np.mean(relative_errors) * 100
    return mape


# Calcular o MAPE
mape_result = calculate_mape(housing_labels, housing_predictions)

# Imprimir o resultado
print(f"O MAPE é: {mape_result:.2f}%")

# QUE TAL TENTAR OUTROS MODELOS?
from sklearn.tree import DecisionTreeRegressor
# Criando o modelo de DecisionTreeRegressor
model_dtr = DecisionTreeRegressor(max_depth=10)  # Número de ramificações da árvore, atenção para não gerar overfitting
model_dtr.fit(housing_prepared, housing_labels)

# vamos tentar o pipeline de pré-processamento completo em algumas instâncias de treinamento
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
predictions = model_dtr.predict(some_data_prepared)
print("Predictions:", model_dtr.predict(some_data_prepared))

print("Labels:", list(some_labels))

# mean_squared_error
housing_predictions = model_dtr.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# mean_absolute_error
lin_mae = mean_absolute_error(housing_labels, housing_predictions)
print(lin_mae)

r2 = r2_score(housing_labels, housing_predictions)
print('r²', r2)

# Calcular o MAPE
mape_result = calculate_mape(housing_labels, housing_predictions)

# Imprimir o resultado
print(f"O MAPE é: {mape_result:.2f}%")
