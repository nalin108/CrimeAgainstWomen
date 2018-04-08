# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 12:05:41 2018

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
#################################################################################################

def ExecuteKMeans(type):
    df =pd.read_excel('in/HouseHold_Clustered.xlsx')
    df = df.loc[df['Type'] == type]
    dataset = df[['Households with electricity (%)', 'Households with an improved drinking-water source1 (%)', 'Households using improved sanitation facility2 (%)','Households using clean fuel for cooking3 (%)','Households with any usual member covered by a health scheme or health insurance (%)']]
    
    dataset.loc[:,'Households with electricity (%)'] *= 0.6
    dataset.loc[:,'Households with an improved drinking-water source1 (%)'] *= 0.5
    dataset.loc[:,'Households using clean fuel for cooking3 (%)'] *= 0.8
    
    kmeans = KMeans(n_clusters=3, random_state=0).fit(dataset.values)
    
    x = kmeans.labels_
    y = dataset.values[:,0]
    ScatterPlot(x,y,'class','Households with electricity (%)')
    y = dataset.values[:,1]
    ScatterPlot(x,y,'class', 'Households with an improved drinking-water source1 (%)')
    y = dataset.values[:,2]
    ScatterPlot(x,y,'class','Households using improved sanitation facility2 (%)')
    y = dataset.values[:,3]
    ScatterPlot(x,y,'class','Households using clean fuel for cooking3 (%)')
    y = dataset.values[:,4]
    ScatterPlot(x,y,'class', 'Households with any usual member covered by a health scheme or health insurance (%)')
    
    dataset= dataset.assign(ClusterClass=pd.Series(kmeans.labels_[:], index = dataset.index))
    dataset = dataset.assign(score = pd.Series(index = dataset.index))
    dataset = dataset.assign(state = pd.Series(index = dataset.index))
    
    for index, row in dataset.iterrows():
        dataset.loc[index,'score']= (dataset.loc[index,'Households using clean fuel for cooking3 (%)']/100 + dataset.loc[index,'Households using improved sanitation facility2 (%)']/100 + dataset.loc[index,'Households with an improved drinking-water source1 (%)']/100 + dataset.loc[index, 'Households with electricity (%)']/100 + dataset.loc[index,'Households with any usual member covered by a health scheme or health insurance (%)']/100)/5
        dataset.loc[index,'state'] = df.loc[index,'District']
    
    dataset.to_csv('out/outHousehold_'+type+'.csv')
########################################################################################################

ExecuteKMeans('rural')
ExecuteKMeans('Urban')
ExecuteKMeans('total')


