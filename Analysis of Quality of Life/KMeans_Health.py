# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:04:06 2018

@author: nalinsharma
"""
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

def ScatterPlot(x,y,xLabel, yLabel):
    plt.scatter(x,y)
    plt.ylabel(yLabel)
    plt.xlabel(xLabel)
    plt.show()

###########################################################################################################

def ExecuteKMeans(type):
    df =pd.read_excel('in/Health_Clustered.xlsx')
    df = df.loc[df['Type'] == type]
    #print (df.tail(1))
    
    dataset = df.drop(['Type', 'District', 'State'],axis=1)
    
    dataset  =dataset.loc[dataset[dataset.columns[9]] != 0]
    dataset  =dataset.loc[dataset[dataset.columns[10]] != 0]
    
    dataset.loc[:,dataset.columns[4]] *= 0.5
    dataset.loc[:,dataset.columns[6]] *= 0.3
    dataset.loc[:,dataset.columns[7]] *= 5
    dataset.loc[:,dataset.columns[8]] *= 5
    dataset.loc[:,dataset.columns[9]] *= 10
    dataset.loc[:,dataset.columns[10]] *= 10
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset.values)
    x = kmeans.labels_
    for i in range(len(dataset.columns)):
        y = dataset.values[:,i]
        ScatterPlot(x,y,'class',dataset.columns[i])

    dataset= dataset.assign(ClusterClass=pd.Series(kmeans.labels_[:], index = dataset.index))
    dataset = dataset.assign(score = pd.Series(index = dataset.index))
    dataset = dataset.assign(state = pd.Series(index = dataset.index))
    
    for index, row in dataset.iterrows():
        dataset.loc[index,'score']= (dataset.loc[index,dataset.columns[0]]/100 + dataset.loc[index,dataset.columns[1]]/100 + dataset.loc[index,dataset.columns[2]]/100 + dataset.loc[index,dataset.columns[3]]/100 + dataset.loc[index,dataset.columns[4]]/100 + dataset.loc[index,dataset.columns[5]]/100 + dataset.loc[index,dataset.columns[6]]/100 + dataset.loc[index,dataset.columns[7]]/100 +  dataset.loc[index,dataset.columns[8]]/100 + dataset.loc[index,dataset.columns[9]]/100 + dataset.loc[index,dataset.columns[10]]/100)/11
        dataset.loc[index,'state'] = df.loc[index,'District']
    
    dataset.to_csv('out/outHealth_'+type+'.csv')
################################################################################################################

ExecuteKMeans('rural')
ExecuteKMeans('total')
ExecuteKMeans('urban')

















