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

pd.set_option('display.max_columns', None)

# Vamos carregar o arquivo csv em nosso drive e analisarmos o nosso dataframe
df_fifa = pd.read_csv("players_22.csv")
print(df_fifa)