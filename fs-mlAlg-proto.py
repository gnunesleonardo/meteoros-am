import csv
import pandas as pd
from pandas import *
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt
import scikitplot as skplt

import os
os.chdir('/home/sramarocas/unifesp/meteoros-am')

######################################################################
## Finalizacao do processamento dos conjuntos de dados ##
######################################################################
## importa conjunto de treino ##
train_data = pd.read_csv('train_df.csv')

## importa o conjunto de teste ##
test_data = pd.read_csv('test_df.csv')

## importa o gabarito do conjunto de treino ##
train_class = pd.read_csv('publ_class.csv')
train_class = train_class.drop('Id', axis=1)
train_class = train_class.drop('Usage', axis=1)
train_class = train_class['Prediction'].tolist()

## importa o gabarito do conjunto de teste ##
test_class = pd.read_csv('priv_class.csv')
test_class = test_class.drop('Id', axis=1)
test_class = test_class.drop('Usage', axis=1)
test_class = test_class['Prediction'].tolist()

######################################################################

######################################################################
## Selecao de atributos baseada em arvores de decisao ##
######################################################################
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

tfs = ExtraTreesClassifier()
tfs = tfs.fit(train_data, train_class)

model = SelectFromModel(tfs, prefit=True)

tfs_train = train_data.loc[:, model.get_support()]
tfs_test = test_data.loc[:, model.get_support()]

tfs_train.shape
tfs_test.shape

######################################################################

######################################################################
## Selecao de atributos recursiva com cross-validation ##
######################################################################
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

## cross-validation por svm ##
svc = SVC(kernel="linear")

## normaliza conjunto de treino ##
min_max_scaler = preprocessing.MinMaxScaler()
train_norm = min_max_scaler.fit_transform(train_data)

## stratified 10-fold cross-validation com svm ##
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(10), scoring='accuracy')
rfecv.fit(train_norm, train_class)

rfe_train = train_data.loc[:, rfecv.get_support()]
rfe_test = test_data.loc[:, rfecv.get_support()]

rfe_train.shape
rfe_test.shape

print("Numero otimizado de atributos: %d" % rfecv.n_features_)

## grafico num. atributos por taxa classif. corretas ##
plt.figure()
plt.xlabel("Numero de atributos")
plt.ylabel("Taxa de classificacoes corretas\ncom cross-validation")
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_, color='red')
plt.savefig('rfe_grafico_atributos.png')
plt.show()

######################################################################

## normalizacao dos conjuntos para o knn, rede neural e rede bayesiana ##
min_max_scaler = preprocessing.MinMaxScaler()
train_final = min_max_scaler.fit_transform(train_data)
test_final = min_max_scaler.fit_transform(test_data)

## conjunto para arvore de decisao ##
dtree_train = train_data
dtree_test = test_data

######################################################################
## PCA Plotting ##
######################################################################
from sklearn.decomposition import PCA as sklearnPCA

X = train_final
train_raw = pd.read_csv('train_raw.csv')
y = train_raw['Class']

## 2-dimensional PCA ##
pca = sklearnPCA(n_components=2)
pca_X = pd.DataFrame(pca.fit_transform(X))

plt.scatter(pca_X[y==0][0], pca_X[y==0][1], label='non-meteor', color='darkorange')
plt.scatter(pca_X[y==1][0], pca_X[y==1][1], label='meteor', color='navy')

plt.legend(loc='lower left', shadow=True, scatterpoints=3)
plt.grid()
plt.savefig('pltdataframe_train.png')
plt.show()

######################################################################

######################################################################
## Algortimo KNN com melhor K encontrado por cross-validation ##
######################################################################
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

## cria uma lista de K vizinhos impares ##
k_vizinhos = list(range(1, 39, 2))

## lista com os valores de predicao ##
pred_list = []

## executa a 10-fold cross validation ##
for k in k_vizinhos:
    knn = KNeighborsClassifier(n_neighbors=k, p=2, metric='euclidean')
    predcross = cross_val_score(knn, train_final, train_class, cv=10, scoring='accuracy')
    pred_list.append(predcross.mean())

tx_erro = [1 - x for x in pred_list]

## determinando o melhor K ##
melhor_k = k_vizinhos[tx_erro.index(min(tx_erro))]
menor_erro = min(tx_erro)
print ('Melhor K encontrado: %d' %melhor_k)
print ('Taxa de erro: %.5f' %menor_erro)

## grafico de taxa de erro pelo numero de vizinhos ##
plt.plot(k_vizinhos, tx_erro, color='red')
plt.xlabel('Numero k de vizinhos')
plt.ylabel('Taxa de erro na classificao')
#plt.savefig('kotimizado_crossvali_tfs.png')
plt.show()

## classifica conjunto de teste com o melhor K encontrado ##
knn = KNeighborsClassifier(n_neighbors=melhor_k, p=2, metric='euclidean')
knn.fit(train_final, train_class)
knn_pred = knn.predict(test_final)
knn_proba = knn.predict_proba(test_final)

## calculo da acuracia ##
knn_score = accuracy_score(test_class, knn_pred)
print ("Acuracia: %.5f" %knn_score)

## perda logaritmica ##
knn_logloss = log_loss(test_class, knn_pred)
print ("Log Loss: %.5f" %knn_logloss)

## calculo da precisao ##
knn_prec = precision_score(test_class, knn_pred)
print ("Precisao: %.5f" %knn_prec)

## matriz de confusao ##
knn_matrix = confusion_matrix(test_class, knn_pred)
print ("Matriz de confusao: ")
print (knn_matrix)

## plot da curva roc ##
skplt.metrics.plot_roc(test_class, knn_proba)
plt.savefig('curvaROC-knnBruto.png')
plt.show()

## plot grafico da matriz de confusao ##
skplt.metrics.plot_confusion_matrix(test_class, knn_pred, normalize=True)
plt.savefig('matrizConfusao-knnBruto.png')
plt.show()

######################################################################

######################################################################
## Rede Neural Artificial (Multi-Layer Perceptron) ##
######################################################################
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)

mlp.fit(train_final, train_class)
mlp_pred = mlp.predict(test_final)
mlp_proba = mlp.predict_proba(test_final)

## calculo da acuracia ##
mlp_score = accuracy_score(test_class, mlp_pred)
print ("Acuracia: %.5f" %mlp_score)

## perda logaritmica ##
mlp_logloss = log_loss(test_class, mlp_pred)
print ("Log Loss: %.5f" %mlp_logloss)

## calculo da precisao ##
mlp_prec = precision_score(test_class, mlp_pred)
print ("Precisao: %.5f" %mlp_prec)

## matriz de confusao ##
mlp_matrix = confusion_matrix(test_class, mlp_pred)
print ("Matriz de confusao: ")
print (mlp_matrix)

## plot da curva roc ##
skplt.metrics.plot_roc(test_class, mlp_proba)
plt.savefig('curvaROC-mlpBruto.png')
plt.show()

## plot grafico da matriz de confusao ##
skplt.metrics.plot_confusion_matrix(test_class, mlp_pred, normalize=True)
plt.savefig('matrizConfusao-mlpBruto.png')
plt.show()

######################################################################

######################################################################
## Rede Bayesiana ##
######################################################################
from sklearn.naive_bayes import GaussianNB

rbayes = GaussianNB()
rbayes.fit(train_final, train_class)

rbayes_pred = rbayes.predict(test_final)
rbayes_proba = rbayes.predict_proba(test_final)

## calculo da acuracia ##
rbayes_score = accuracy_score(test_class, rbayes_pred)
print ("Acuracia: %.5f" %rbayes_score)

## perda logaritmica ##
rbayes_logloss = log_loss(test_class, rbayes_pred)
print ("Log Loss: %.5f" %rbayes_logloss)

## calculo da precisao ##
rbayes_prec = precision_score(test_class, rbayes_pred)
print ("Precisao: %.5f" %rbayes_prec)

## matriz de confusao ##
rbayes_matrix = confusion_matrix(test_class, rbayes_pred)
print ("Matriz de confusao: ")
print (rbayes_matrix)

## plot da curva roc ##
skplt.metrics.plot_roc(test_class, rbayes_proba)
plt.savefig('curvaROC-rbayesBruto.png')
plt.show()

## plot grafico da matriz de confusao ##
skplt.metrics.plot_confusion_matrix(test_class, rbayes_pred, normalize=True)
plt.savefig('matrizConfusao-rbayesBruto.png')
plt.show()

######################################################################

######################################################################
## Arvore de Decisao ##
######################################################################
from sklearn import tree

dtree = tree.DecisionTreeClassifier()
dtree = dtree.fit(dtree_train, train_class)

dtree_pred = dtree.predict(dtree_test)
dtree_proba = dtree.predict_proba(test_final)

## calculo da acuracia ##
dtree_score = accuracy_score(test_class, dtree_pred)
print ("Acuracia: %.5f" %dtree_score)

## perda logaritmica ##
dtree_logloss = log_loss(test_class, dtree_pred)
print ("Log Loss: %.5f" %dtree_logloss)

## calculo da precisao ##
dtree_prec = precision_score(test_class, dtree_pred)
print ("Precisao: %.5f" %dtree_prec)

## matriz de confusao ##
dtree_matrix = confusion_matrix(test_class, dtree_pred)
print ("Matriz de confusao: ")
print (dtree_matrix)

## plot da curva roc ##
skplt.metrics.plot_roc(test_class, dtree_proba)
plt.savefig('curvaROC-dtreeBruto.png')
plt.show()

## plot grafico da matriz de confusao ##
skplt.metrics.plot_confusion_matrix(test_class, dtree_pred, normalize=True)
plt.savefig('matrizConfusao-dtreeBruto.png')
plt.show()

######################################################################
