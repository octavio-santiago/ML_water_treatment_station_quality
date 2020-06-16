# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 16:46:04 2019

@author: Octavio
"""

import pandas as pd
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt

#importacao do dataset
df = pd.read_csv('DS4.csv',sep=',')

#pre tratamento
df = df.drop(columns=['Afluente','LZAM','Sanitario','Aeracao','Vol Eq','Vol Aera','Desenquadrou'])
df = df.replace('#N/D',pd.np.nan)
                
                
df['phEntrada']=df['phEntrada'].astype(float)
df['phNeutralizacao']=df['phNeutralizacao'].astype(float)
df['phEqualizacao']=df['phEqualizacao'].astype(float)
df['phAeracao']=df['phAeracao'].astype(float)
df['OD']=df['OD'].astype(float)
df['VLodoSD30']=df['VLodoSD30'].astype(float)   
df['VAeracao']=df['VAeracao'].astype(float)  
df['VEqualizacao']=df['VEqualizacao'].astype(float)   

#limpeza dos nas
df = df.dropna(subset=['Equalizacao','Eq D-1','phEntrada','Efluente Final'])
df['SAO'] = df['SAO'].fillna(method='ffill')
df = df.reset_index(drop=True)

df.to_csv('dfPreTratado.csv', sep=',', encoding='utf-8')
#modeificacao
df = df.drop(columns=['Equalizacao','VAeracao','VEqualizacao','SAO'])

#preparacao para treino
predictors = df.loc[:,"Eq D-1":"VLodoSD30"]
response = df['Efluente Final']

X_train, X_test, Y_train, Y_test = train_test_split(predictors, response, test_size=0.33, random_state=42)


modelos = [linear_model.SGDRegressor(),linear_model.LinearRegression(),linear_model.Lasso(alpha=0.1),
           linear_model.LassoCV(cv=10),linear_model.LassoLarsIC(criterion='bic'),linear_model.LassoLars(alpha=.1),
           linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0], cv=3),svm.SVR(),linear_model.BayesianRidge(),
           linear_model.SGDRegressor(max_iter=1000, tol=1e-3)]

modelosList = ['SGDRegressor','LinearRegression','Lasso','LassoCV','LassoBIC','LARS Lasso','Ridge','SVM','Bayesian Ridge','SGDREG']


#modelo manual
for i,val in enumerate(modelosList):
    print (i,'-',val)
    
mod = input("Escolha o modelo: ")

reg = modelos[int(mod)]

reg.fit(X_train, Y_train)  

teste = pd.DataFrame(reg.predict(X_test))
#score = reg.score(X_train, Y_train)
score = reg.score(X_test,Y_test)
print(modelosList[int(mod)],'Score: ',score)

plt.scatter(teste,Y_test)

#teste = teste.merge(df['Efluente Final'])
#teste = pd.concat([teste, Y_test],axis=1,ignore_index=True)
          
#modelo automatico

for i,val in enumerate(modelosList):
    reg = modelos[int(i)]
    reg.fit(X_train, Y_train) 
    teste = pd.DataFrame(reg.predict(X_test))
    score = reg.score(X_test,Y_test)
    if i == 0:
        MaxScore = score
        idx = i
    else:
        if score > MaxScore:
            MaxScore = score
            idx = i
            
        
    
print("The best model is: ",modelosList[int(idx)])
print(modelosList[int(idx)],'Score: ',MaxScore)

plt.scatter(teste,Y_test)
       
             