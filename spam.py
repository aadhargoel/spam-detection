# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 18:19:46 2019

@author: AADHAR.GOEL

"""

from __future__ import print_function , division
from future.utils import iteritems
from builtins import range
#pip install -U future   #Updating version of future

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes  import MultinomialNB
import os
import matplotlib.pyplot as plt

pip install wordcloud  #Installing wordcloud
from wordcloud import WordCloud,STOPWORDS 

df = pd.read_csv(r'D:\CourseDS\PythonProjects\UdemyNLP\spam.csv', encoding='ISO-8859-1') 
 #UTF-* encoding errors out because of invalid characters these days that includes emojis et so using ISO
df[:1]


df2 = pd.read_csv(r'D:\CourseDS\PythonProjects\UdemyNLP\spam.csv')

df.columns

#We  only need v1 and V2 columns

df=df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1)

# Renaming the reaminging columns as labels and data

df.columns= ['labels', 'data']

#mapping ham and spam to 0 and 1 in labels colums
df['b_labels'] = df['labels'].map({'ham':0 , 'spam':1})
df['b_labels']
df.loc[1,:]
Y=df['b_labels'].as_matrix()
Y
len(df)
len(df[df.labels=='spam'])   ##Gives the count how many spam are there i.e 747

#Converting the words inot numbers using CountVectorizer (It counts the number of words in the document)
count_vectorizer = CountVectorizer(decode_error = 'ignore')

X = count_vectorizer.fit_transform(df['data'])
X.toarray() #Converted the count to an array(Suppose are is repeated twice in document do it will put valu 2 to mapping column to are)
count_vectorizer.get_feature_names() #it will get the names. So say are is repeated twice in document so it
#####We can also use the TfidfVectorizer for the same calculation

###Splitting up data ,, we can also use cross validation to check ho well model does

Xtrain,Xtest, Ytrain, Ytest = train_test_split(X,Y,test_size = 0.33)

""" There are 3 types of Naive Bayes
1) Multinomial - When your featres(Categorical or continuous) describe discrete fequency counts (eg.WordCount)
2)bernoulli - Good for making predictions from binary features
3)Gaussian - good for making predictions from normally distributed features
"""

model = MultinomialNB()
model.fit(Xtrain,Ytrain)
print("train score" , model.score(Xtrain,Ytrain))
print("train score" , model.score(Xtest,Ytest))



########VISUALIZING THE DATA##############
stopword = set(STOPWORDS)
def visualize(label):
        words =''
        for msg in df[df['labels']==label]['data']:
            msg=msg.lower()
            words +=msg+''
        wordcloud = WordCloud(width = 800, height = 800, 
                background_color ='white', 
                stopwords = stopword, 
                min_font_size = 10).generate(words)
        plt.figure(figsize = (8, 8), facecolor = None) 
        plt.imshow(wordcloud)
        plt.axis('off')
        plt.show()
     
 visualize('spam')   
 visualize('ham')   
 
 #See what are we getting wrong
 df['predictions'] = model.predict(X)
        
#checking the meesages that should be spam but are not predicted correctly
 
 sneaky_spam = df[(df['predictions']==1) & (df['labels']==0)]['data']
sneaky_spam
for msg in sneaky_spam:
    print(msg)
