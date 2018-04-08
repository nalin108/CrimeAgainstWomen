# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 05:49:06 2018

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
#########################################################################################################################
def ExecuteKMeans(type):
    df =pd.read_excel('in/Women_Empowerment.xlsx')
    df = df.loc[df['Type'] == type]
    
    dataset = df.drop(['Type', 'State'],axis=1)
    dataset.loc[:,'Ever-married women who have ever experienced spousal violence (%)'] *=-1
    dataset.loc[:,'Ever-married women who have experienced violence during any pregnancy (%)'] *=-1
    dataset.loc[:,'Women owning a house and/or land (alone or jointly with others) (%)'] *= 5
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset.values)
    x = kmeans.labels_
    for i in range(len(dataset.columns)):
        y = dataset.values[:,i]
        ScatterPlot(x,y,'class',dataset.columns[i])
    
    dataset= dataset.assign(ClusterClass=pd.Series(kmeans.labels_[:], index = dataset.index))
    dataset = dataset.assign(score = pd.Series(index = dataset.index))
    dataset = dataset.assign(state = pd.Series(index = dataset.index))
    
    for index, row in dataset.iterrows():
        sum=0
        for i in range(len(dataset.columns) -2):
            sum += dataset.loc[index,dataset.columns[i]]/100
        dataset.loc[index,'score'] = sum/len(dataset.columns)
        dataset.loc[index,'state'] = df.loc[index,'State']
    
    dataset.to_csv('out/outWomenEmp_'+type+'.csv')
###############################################################################################################################

ExecuteKMeans('Urban')
ExecuteKMeans('Rural')

