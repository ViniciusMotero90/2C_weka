import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

# Carregando o dataset Iris
arquivo = pd.read_csv('column_2C_weka.csv')
faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull()/len(arquivo['pelvic_incidence']))*100
arquivo['class'] = arquivo['class'].replace('Abnormal',1)
arquivo['class'] = arquivo['class'].replace('Normal',0)
y = arquivo['class']
x = arquivo.drop('class',axis=1)

# Definindo os parâmetros para GridSearchCV
minimo_split = np.array([2, 3, 4, 5, 6, 7, 8])
maximo_nivel = np.array([3, 4, 5, 6])
algoritmo = ['gini', 'entropy']
valores_grid = {
    'min_samples_split': minimo_split,
    'max_depth': maximo_nivel,
    'criterion': algoritmo
}

# Treinando o modelo usando GridSearchCV
modelo = DecisionTreeClassifier()
gridDecisionTree = GridSearchCV(estimator=modelo, param_grid=valores_grid, cv=5)
gridDecisionTree.fit(x, y)

# Resultados do melhor modelo
melhor_modelo = gridDecisionTree.best_estimator_
print('Mínimo split:', melhor_modelo.min_samples_split)
print('Máximo profundidade:', melhor_modelo.max_depth)
print('Algoritmo escolhido:', melhor_modelo.criterion)
print('Acurácia:', gridDecisionTree.best_score_)

# Visualizando a árvore de decisão final
plt.figure(figsize=(12, 8))
plot_tree(
    melhor_modelo,
    filled=True
)
plt.show()
