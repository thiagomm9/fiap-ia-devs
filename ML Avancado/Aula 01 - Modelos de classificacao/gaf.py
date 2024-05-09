
import pandas as pd

dados = pd.read_excel('gaf_esp.xlsx')

print(dados.head())

print(dados.tail())

print(dados.describe())

print(dados.shape)

print(dados.groupby('Espécie').describe())

dados.plot.scatter(x='Comprimento do Abdômen', y='Comprimento das Antenas')

