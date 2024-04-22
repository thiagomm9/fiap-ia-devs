#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 17:09:56 2021

@author: rafaeldontalgoncalez
"""

######################################
# Importando as libraries
######################################

import pandas as pd
import sklearn.model_selection as ms
import sklearn.linear_model as lm
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import os

pd.set_option('display.max_columns', None)  # Para mostrar todas as colunas

######################################
# Importa o dataset
######################################


dataset = pd.read_csv(os.path.dirname(os.path.abspath(
    __file__)) + "/2020_Data_Professional_Salary_MultipleFeatures.csv")
dataset = dataset.dropna()  # dropa os valores nulos
# coluna 0(primeira) ate a -2 (antepenultima) / variaveis descritivas(features)
X = dataset.iloc[:, :-2]
y = dataset.iloc[:, -1].values  # somente a ultima coluna

######################################
# Codificando variaveis Dummy
######################################
X_dummies = pd.get_dummies(X)
# Transforma as variaveis categoricas em dummies(cada categoria vira uma variavel binaria)
# O modelo de regressao linear multipla so aceita variaveis numericas

######################################
# Separar dados em Treino e Teste
######################################

# X_train, X_test, y_train, y_test = ms.train_test_split(X_dummies, y, test_size=1 / 5, random_state=0)
X_train, X_test, y_train, y_test = ms.train_test_split(
    X_dummies, y, test_size=0.2, random_state=0)

######################################
# Treinando o modelo
######################################

regressor = lm.LinearRegression()
regressor.fit(X_train, y_train)

######################################
# Previsao
######################################
y_pred = regressor.predict(X_test)


np.set_printoptions(precision=2)
result = np.concatenate((y_pred.reshape(len(y_pred), 1),
                        y_test.reshape(len(y_test), 1)), 1)


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df


X_reverse = undummify(X_test)
X_reverse = X_reverse.reset_index(drop=True)

y_compare = pd.DataFrame(result)
y_compare = y_compare.rename(index=str, columns={0: 'y_pred', 1: 'y_test'})
y_compare = y_compare.reset_index(drop=True)

resultado_final = pd.concat([y_compare, X_reverse], axis=1)


######################################
# Valor Especifico
######################################

print(regressor.predict([[10]]))
