# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 16:36:22 2020

@author: Trisha
"""
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
from time import time
import pandas as pd
import seaborn as sns

#%% Plot yearly trends
def YearlyTrendsPlot(dSup, P):
    avgP = np.mean(P, axis=1) #np.ones([(2020-1996),1])*
    
    Years = range(1996,2020)
    # order = [0,3,4,9,11,1,2,5,6,7,8,10,12,13,14]
    fig, ax = plt.subplots(1,3,figsize=(15,5),tight_layout = True)
    start = 0
    for s in range(3):
        end = start + 5
        for i in range(start,end):
            if i==1 or i==3 or i==7 or i==9 or i==6 or i==0:
                ax[s].plot(Years,P[:,i],'-',label=str(i)+' '+str(dSup.topWords[i][:3]), lw = 2)
            else:
                ax[s].plot(Years,P[:,i],'-',label=str(i)+' '+str(dSup.topWords[i][:1]),alpha = 0.3, lw = 2)
        ax[s].plot(Years,avgP,'--')
        ax[s].set_xlabel('Year', fontsize=15)
        ax[s].set_ylabel('Topic Popularity', fontsize=15)
        ax[s].set_ylim([0, 0.4])
        ax[s].legend(loc='lower left', bbox_to_anchor= (0.0, 1.01), ncol=1, 
                borderaxespad=0, frameon=False, fontsize=12)
        start = end
    plt.show()

#%% WOrdcloud for overall topics
def TopicWordClout(dSup):
    # Create a list of word
    text = []
    freq = []
    for i in range(len(dSup)):
        for j in range(3):
            text.append(dSup.loc[i].topWords[j])
            freq.append(float(dSup.loc[i].Popularity))
            
    freq = np.hstack(freq)
    text = np.hstack(text)
    dictionary = dict(zip(text, freq))
    
    # Create the wordcloud object
    wordcloud = WordCloud(width=480*3, height=480*2, margin=0, colormap='autumn',max_font_size=150, min_font_size=30,
                          background_color='black').fit_words(dictionary)
    
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=10, y=10)
    plt.show()
    
#%% Plot the number of authors per paper
def CollaborationPlot(dfPlot):    
    dfPlot['#Authors'] = dfPlot['#Authors'].astype(float)
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(x="Year", y="#Authors",
                  data=dfPlot, showfliers=False)
    plt.xlabel('Year', fontsize=18)
    plt.ylabel('Authors / Paper', fontsize=18)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=18)
    plt.show()
    
