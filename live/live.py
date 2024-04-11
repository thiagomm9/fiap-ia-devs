import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None) # Para mostrar todas as colunas
df = pd.read_excel("StudentsPrepared.xlsx")
df.head()

df.shape

df.isnull().sum()

set(df['Target'])

#Calcular o total de alunos(as) por tipo de status
df_targets_percent = df.groupby('Target')['Target'].count()
# Calculando a proporção de estudantes por categporia
total_estudantes = len(df)
df_target_porcentagem = df_targets_percent / total_estudantes * 100
#Separando os valores e nomes em uma lista, para deixar o gráfico mais apresentável
labels = df_target_porcentagem.index.tolist()
sizes = df_target_porcentagem.values.tolist()
#Criando o gráfico
plt.style.use('dark_background')
figura, grafico = plt.subplots(figsize=(18, 6))
grafico.pie(sizes, autopct='%1.0f%%', colors=[ '#e34c42','#4dc471','#3b71db'], labeldistance = 1.1, explode=[0, 0, .4])
grafico.axis('equal')
plt.title('Porcentagem por status target')
plt.legend(labels, loc='best')
plt.show()

df_evadidos = df[df['Target'] == 'Desistente']
df_matriculados = df[df['Target'] == 'Matriculado']
df_graduados = df[df['Target'] == 'Graduado']
df_concatenado = pd.concat([df_evadidos, df_graduados])
set(df_concatenado['Target'])