# Import necessary libs
import pandas as pd  # leitura de dados
import matplotlib.pyplot as plt  # graficos

from sklearn.model_selection import train_test_split  # separar dados em treino e teste
from sklearn.linear_model import LinearRegression  # ALgoritmo de regressão linear
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Métricas de validação do modelo

# Import data base
dados = pd.read_excel('Sorvete.xlsx')  # importa base xlsx
print(dados.head(), '\n')  # Imprime as 5 primeiras linhas

# Visualize data, graphic plot
# Cria um gráfico de dispersão(scatter no matplotlib)
plt.scatter(dados['Temperatura'], dados['Vendas_Sorvetes'])
# Personalização do gráfico (nomes(labels))
plt.xlabel('Temperatura (ºC)')
plt.ylabel('Venda de Sorvetes (milhares)')
plt.title('Relação entre Temperatura e Venda de Sorvetes')
# Exibição
plt.show()

# Exibe a correlação
print(dados.corr(), '\n')

### CRIANDO O MODELO DE REGRESSÃO LINEAR

# Dividindo os dados em conjuntos de treinamento e teste
X = dados[['Temperatura']]  # Recurso (variável indepentente)
y = dados['Vendas_Sorvetes']  # Rótulo (variável dependente)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Método do sklearn
# test size define o tamanho da base, no caso 20% para base de teste e 80% para treinar o modelo
# No caso do random_state, ele é uma semente aleatória,
# em que um pedaço do conjunto de dados treino e teste será mantido estável toda vez que rodar o modelo.
# sem ele pode ser que a base seja distribuida entre treino e teste
# de forma aleatoria, e gerar dados diferentes, o que pode acabar enviesando o modelo e não performar corretamente

# Retorna o tamanho da base de dados (quantidade de linhas(registros) e colunas)
print(X_train.shape, '\n')
print(X_test.shape, '\n')

# Criando e treinando o modelo de regressão linear simples
modelo = LinearRegression()  # Aqui podem ser alterado os valores para a expressão da regressão linera, possui valores default
modelo.fit(X_train, y_train)  # Treinar o modelo

# Fazendo previsões(y) no conjunto de teste
previsoes = modelo.predict(X_test)  # Apenas na base de teste(dados não vistos antes)
print(previsoes, '\n')

###AVALIAÇÃO DE RESULTADOS

# Avaliando o desempenho do modelo
# *** O RMSE é a raiz quadrada do MSE (Erro Quadrático Médio - Mean Squared Error).
# o MSE é a média dos quadrados das diferenças entre os valores reais e os valores previstos.
# *** O MAE (Erro Médio Absoluto - Mean Absolute Error), onde um valor pequeno para MAE significa
# que suas previsões estão próximas das reais.
# *** O "R-squared (R2)" fornece informações sobre o ajuste geral do modelo.
# O valor do R2 pode variar entre 0 e 1, quanto mais próximo de 1, melhor,
# pois indica que o modelo explica uma maior proporção da variabilidade nos dados.
# O R2 é também uma das principais métricas de avaliação do modelo de regressão.
erro_medio_quadratico = mean_squared_error(y_test, previsoes)
erro_absoluto_medio = mean_absolute_error(y_test, previsoes)
r_quadrado = r2_score(y_test, previsoes)
print(f'Erro Médio Quadrático: {erro_medio_quadratico}')
print(f'Erro Absoluto Médio: {erro_absoluto_medio}')
print(f'R² (coeficiente de determinação): {r_quadrado}')

# Visualizando as previsões com plot de gráfico
plt.scatter(X_test, y_test, label='Real')  # gráfico de dispersão com os dados reais
plt.scatter(X_test, previsoes, label='Previsto', color='red')  # previsão
plt.xlabel('Temperatura (°C)')
plt.ylabel('Vendas de Sorvetes (milhares)')
plt.title('Previsões do Modelo de Regressão Linear')
plt.legend()
plt.show()
