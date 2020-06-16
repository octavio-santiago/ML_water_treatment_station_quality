# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:48:23 2019

@author: Octavio
"""

import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#importacao do dataset
df = pd.read_csv('DS4.csv',sep=',')

#esparçamento dos dados
plt.hist(df['Efluente Final'].dropna(),bins=20)    
plt.xlabel('DQO mg/L')     
plt.ylabel('Distribuição') 
plt.title('Distribuição do Efluente final') 
plt.xticks([0,170,250,400,500,700,800,1000,1200,1400])
plt.grid(True)
plt.show()
plt.clf()
plt.hist(df['Eq D-1'].dropna(),bins=20)  
plt.xlabel('DQO mg/L')     
plt.ylabel('Distribuição')  
plt.title('Distribuição do Afluente') 
plt.grid(True) 
plt.show()
plt.clf()
plt.hist(df['Desenquadrou'].dropna(),bins=20)  
plt.xlabel('Desenquadrou')     
plt.ylabel('Distribuição')  
plt.title('Distribuição do Desenquadramento') 
plt.grid(True) 
plt.show()
plt.clf()
plt.scatter(df['Eq D-1'],df['Desenquadrou'])
plt.ylabel('Desenquadrou')     
plt.xlabel('Efluente D-1')  
plt.title('Distribuição do Desenquadramento') 
plt.grid(True) 
plt.show()
plt.clf()

#pre tratamento
df = df.drop(columns=['Afluente','LZAM','Sanitario','Aeracao','Vol Eq','Vol Aera','Efluente Final'])
df = df.replace('#N/D',pd.np.nan)

   
                
df['phEntrada']=df['phEntrada'].astype(float)
df['phNeutralizacao']=df['phNeutralizacao'].astype(float)
df['phEqualizacao']=df['phEqualizacao'].astype(float)
df['phAeracao']=df['phAeracao'].astype(float)
df['OD']=df['OD'].astype(float)
df['VLodoSD30']=df['VLodoSD30'].astype(float)   
df['VAeracao']=df['VAeracao'].astype(float)  
df['VEqualizacao']=df['VEqualizacao'].astype(float)   

plt.hist(df['phEntrada'].dropna(),bins=20)  
plt.xticks([0,2,4,6,7,8,9,10,12])  
plt.xlabel('pH de Entrada')     
plt.ylabel('Distribuição') 
plt.title('Distribuição do pH de entrada') 
plt.grid(True)
plt.show()
plt.clf()

#limpeza dos nas
df = df.dropna(subset=['Equalizacao','Eq D-1','phEntrada','Desenquadrou','phAeracao D-1'])
df['SAO'] = df['SAO'].fillna(method='ffill')
df = df.reset_index(drop=True)


#Usando PCA para reduzir a dimensionalidade
pca = PCA(n_components=2)
pca.fit(df.loc[:,"Eq D-1":"DiasDescarte"])
X = pca.fit_transform(df.loc[:,"Eq D-1":"DiasDescarte"])
print("PCA: ",pca.explained_variance_ratio_)
color = df.loc[:,"Desenquadrou"]
#plt.scatter(df.loc[:,"Data"],X,c=color)
plt.scatter(X[:,0],X[:,1],c=color)
plt.show()
plt.clf()



df.to_csv('dfPreTratado2.csv', sep=',', encoding='utf-8')
#modificacao
df = df.drop(columns=['Equalizacao','VAeracao','VEqualizacao','SAO'])

#preparacao para treino
predictors = df.loc[:,"Eq D-1":"VLodoSD30"]
response = df['Desenquadrou']

X_train, X_test, Y_train, Y_test = train_test_split(predictors, response, test_size=0.30, random_state=42)

clf = SGDClassifier()
clf.fit(X_train, Y_train) 

y_pred = pd.DataFrame(clf.predict(X_test))
#score = reg.score(X_train, Y_train)
score = clf.score(X_test,Y_test)
print('Score: ',score)
matrix = confusion_matrix(Y_test,y_pred)
print(matrix)
print("Acertos %: ",score *100)
print("Erros %: ", (1-score) *100)



#plt.scatter(y_pred,Y_test) 