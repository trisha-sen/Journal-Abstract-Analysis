# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:48:50 2020

@author: Trisha
"""
#%% import modules
from time import time
import pandas as pd
from nltk.tokenize import word_tokenize #, sent_tokenize
from gensim.models import Phrases
#import gensim 
from nltk.corpus import stopwords
import string 
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
#from matplotlib.ticker import MaxNLocator
import warnings 
warnings.filterwarnings(action = 'ignore')

# %% Laoding Data
class loadData:
    def __init__(self,filename):    
        print("Loading dataset...")
    
        t0 = time()
        # df = pd.read_csv('IECR_articles.csv')
        df = pd.read_csv(filename)
    
        for i in df.index:
            if pd.isnull(df.Abstract[i]): #
                df.Abstract[i] = ''
                
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        
        df = df[df['Type']=='Article']
        df.drop(columns=['Unnamed: 0'],inplace = True)
        df.reset_index(inplace = True,drop=True)
    
        print("done in %0.3fs." % (time() - t0))
        
        self.df = df

# %% Tokenizing abstracts
class tokenizeAbstracts:
    def __init__(self,df):
        df = df[df['Type']=='Article']
        word2vecSamples = list(df['Abstract'])
        
        stop_words = set(stopwords.words('english'))
        
        t0 = time()
        data = []
        for i in word2vecSamples:
            temp=[]    
            for j in word_tokenize(i):
                if j.lower() not in stop_words:
        #             if j == 'amino':
        #                 print(j)
                    temp.append(j.lower().translate(str.maketrans('', '', string.punctuation)))
                
            data.append(temp) 
        
        self.data = data
        self.bigram_transformer = Phrases(data)
        
        print("done in %0.3fs." % (time() - t0))
        
        
#%% implementing tf-idf transformation        
def tfidfTansformation(df,n_features):
    t0 = time()
    print("\nExtracting tf-idf features for NMF...")
    
    # Use tf-idf features for NMF.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, 
                                       max_features=n_features,ngram_range=(1, 2),
                                       stop_words='english')
    
    #df = df[df['Type']=='Article']
    #df.drop(columns=['Unnamed: 0'],inplace = True)
    #df.reset_index(inplace = True,drop=True)
    
    tfidf_vectorizer.fit(list(df['Abstract']))
    
    print("done in %0.3fs." % (time() - t0))
    return(tfidf_vectorizer)


#%% Getting top words to describe topic  
def top_words(model, feature_names, n_top_words, P, Year):
    Topic_popular =list(); topWords = list();
    for topic_idx, topic in enumerate(model.components_):
        topWords.append(list([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
        Topic_popular.append(round(P[topic_idx],2))
    d = {'Popularity': Topic_popular} #pd.DataFrame(topWords, Topic_popular)
    df = pd.DataFrame(data=d)
    df['topWords'] = topWords
    df['Year'] = Year
    df.sort_values(by=['Popularity'],inplace = True, ascending=False)
    df.rename(columns={df.columns[0]: "Popularity" }, inplace = True)
    return(df)

def print_top_words(model, feature_names, n_top_words, P):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += str(round(P[topic_idx],2))+" "
        message += " ".join([feature_names[i]+','
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
#%% MNF Factorization per year
class NMFoperation:    
    def __init__(self, df, n_samples, n_features, n_components, n_top_words, tfidf_vectorizer):
        print("\nFitting the NMF model (Frobenius norm) with tf-idf features, "
          "n_samples=%d and n_features=%d..."
          % (n_samples, n_features))
    
        HAll = []; WAll = []
        dtopWords = pd.DataFrame(columns=['Popularity','topWords','Year'])
        dfFinal = pd.DataFrame(columns = df.columns)

        for i in range(1996,2020):
            df0 = df[(df['Year'] == i) & (df['Category']!='Mastheads')]
            
            dfFinal = dfFinal.append(df0, ignore_index = True)            
            # print(i),
            
        #     df0 = df0[df0['Type']=='Article']
            data_samples = list(df0['Abstract'])
            
            tfidf = tfidf_vectorizer.transform(data_samples)
        
            # Fit the NMF model
            nmf_model = NMF(n_components=n_components, random_state=1,
                      alpha=.1, l1_ratio=.5)
            nmf = nmf_model.fit(tfidf)
        
            W = nmf_model.transform(tfidf)
            H = nmf_model.components_
            
            # Popularity Ranking
            P = np.sum(W, axis = 0)/np.sum(W)  
        
            tfidf_feature_names = tfidf_vectorizer.get_feature_names()
        
            topWords = top_words(nmf, tfidf_feature_names, n_top_words, P, i)
            
            HAll.append(H)
            WAll.append(W)
            dtopWords = pd.concat([dtopWords, topWords], ignore_index=True)
            
        self.HAll = HAll
        self.Wall = WAll
        self.dtopWords = dtopWords
        self.dfFinal = dfFinal

#%% 2nd layer of overall NMF factorization
class NMF2ndlayer:
    def __init__(self,HAll,NComp):
        HAll = np.vstack(HAll)
        # print(len(data_samples))
        # print((tfidf.shape))
        #Fit the NMF model
        nmf_model = NMF(n_components=NComp, random_state=1,
                  alpha=.1, l1_ratio=.5)
        
        nmf = nmf_model.fit(HAll)
        self.U = nmf_model.transform(HAll)
        self.L = nmf_model.components_
        
#%% Plotting the similarity between topics
class plotSimilarity:
    def __init__(self,L):
        similarity = np.zeros([L.shape[0],L.shape[0]])
        similarity.fill(np.nan)
        for i in range(L.shape[0]):
            for j in range(0,i):
                similarity[i,j] = np.dot(L[i],L[j])/(np.linalg.norm(L[i])*np.linalg.norm(L[j]))     
        self.similarity = similarity

        fig, ax = plt.subplots(1,1,figsize=(5,5))
        plt.set_cmap('viridis_r')
        im = ax.imshow(similarity,vmin=0, vmax=1)
        plt.title('Topic Similarity')
        fig.colorbar(im, ax=ax)
        plt.show()
        
#%% Getting top words to describe topic
class GetTopWords:
    def __init__(self, n_top_words, U, nmf, tfidf_vectorizer):
        
        tfidf_feature_names = tfidf_vectorizer.get_feature_names();
        P = np.sum(U, axis = 0)/np.sum(U) 
        #n_top_words = 10
        dSup = top_words(nmf, tfidf_feature_names, n_top_words, P, 'All')
        
        #dSup.to_pickle("dSup.pkl")
        self.dSup = dSup

#%% Annual trends in popularity per topic       
class YearlyPopularity:
    def __init__(self,WAll,U):
        # WAll0 = np.vstack(WAll)
        Year = 1996
        P = []
        LHSmega = []
        for i in range(len(WAll)):
            W = WAll[i]
            UYear = U[i*8:(i+1)*8]
            LHS = np.matmul(W,UYear)
            LHSmega.append(LHS)
            Year = Year+1
            # Popularity Ranking
            P.append(np.sum(LHS, axis = 0)/np.sum(LHS))
        
        self.P=np.vstack(P)
        self.LHSmega = LHSmega
        
#%% Calculating citations per year
class Impact:
    def __init__(self, df):
        # year1 = range(1999,2019)
        Years = range(1996,2020)
        totPapers = []
        for i in Years:
            totPapers.append(len(df[df['Year'] == i]))
        self.impactFactor = [1.39, 1.421, 1.423, 1.425, 1.42, 1.643, 1.787, 1.739, 1.907, 1.95, 1.894, 2.391, 2.665, 2.591, 2.633, 2.987, 2.898, 3.104, 3.349, 3.539]
        
        # Impact Factor
        avgTot = []
        year2 = []
        for i in range(1996,2019):
            if i != 2004:
                dfTemp = df[(df.Year == i)]
                Citations = dfTemp['Cited by']
                avgTot.append(round(np.mean(Citations)/(2019-i),2))
                year2.append(i)
            
        self.avgTot = avgTot
        # year2 = np.arange(1996,2019)
        
#%% Which topic does each paper belong to
class LHSmega2ArgVal:
    def __init__(self,LHSmega):
        
        def argFunc(x):
            args = np.flip(np.argsort(x)[-4:])  
            vals = np.flip(np.sort(x)[-4:])
            args[np.argwhere(np.isnan(vals))] = -111
            return args
        def valFunc(x):
            vals = np.flip(np.sort(x)[-4:])
            vals[np.argwhere(np.isnan(vals))] = 0
            return vals
        
        args = []; vals = []
        for i in range(len(LHSmega)):
            LHSsum = np.sum(LHSmega[i], axis = 1)
            args.append(np.apply_along_axis(argFunc, 1, (LHSmega[i]/LHSsum[:,None])))
            vals.append(np.apply_along_axis(valFunc, 1, (LHSmega[i]/LHSsum[:,None])))
        
        self.argsAll = np.vstack(args)
        self.valsAll = np.vstack(vals)

#%% Augmenting citation and corresponding author info to df       
def Add_Citations_Corresponding_Author(dfPlot):      
    def splitName(df):    
        if (pd.isnull(df['Corresponding Author'])) | (df['Corresponding Author']=='NA'):
            x = 'NA'
        else:
            x = word_tokenize(df['Corresponding Author'])
            for i in range(len(x)-1):
                x[i] = x[i][0]
            x = ' '.join(x).rsplit('*')[0]
        return x
    
    def citationsPerYear(df):
        x = round(df['Cited by']/(2020-df['Year']),2)
        return(x)
    
    t0 = time()
    dfPlot['Corresponding Author'] = dfPlot.apply(lambda x: splitName(x), axis = 1)
    dfPlot['Annual Citations'] = dfPlot.apply(lambda x: citationsPerYear(x), axis = 1)
    print("done in %0.3fs." % (time() - t0))        
    
    return(dfPlot)