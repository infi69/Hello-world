# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 18:19:10 2019

@author: anant.jain08
"""

import pandas as pd
import numpy as np
df=pd.read_excel(r'D:\Learning _Documentation\FINAL.xlsx')
#%%
df['Sample1']=df['Sample1'].apply(lambda x:str(x)).fillna(' ')
df['Sample2']=df['Sample2'].apply(lambda x:str(x)).fillna(' ')
df['Sample3']=df['Sample3'].apply(lambda x:str(x)).fillna(' ')
#df['Sample4']=df['Sample4'].fillna(' ')

df['Sample_Values']=df['Sample1']+' '+df['Sample2']+' '+df['Sample3']

#%%
print(df.shape[0])
print(df.columns)
#%%
import re
for i in range(0,df.shape[0]):
    df['Column_name1'][i]=df['Column_name1'][i].lower()
    df['Column_name1'][i]=re.sub('[^A-Za-z0-9]',' ',df['Column_name1'][i])
    df['Sample_Values'][i]=df['Sample_Values'][i].lower()
    df['Sample_Values'][i]=re.sub('[^A-Za-z0-9]',' ',df['Sample_Values'][i])
    
    #%%
print(df.iloc[:,1])
print(df.head())
df['Sample_Values']
#%%
cols=['Column_name1','Sample_Values']
rule_id_df=df['Class']
rule_ids=rule_id_df.drop_duplicates().sort_values()
#%%
df=df[cols]
df.info()
#%%
print(rule_ids)
df['X']=df['Column_name1']+' '+df['Sample_Values']
#%%
df['X'].head()
#%%
df['Column_name1'].head()
#%%
#features analysis
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(sublinear_tf=True,stop_words='english',max_features=1000,norm='l2',encoding='latin-1',ngram_range=(1,2))
features=tfidf.fit_transform(df['X']).toarray()
labels=rule_id_df

for rule_id in rule_ids:
    counts=np.sum(labels==rule_id)
    print("rule id:{}, no of samples:{}".format(rule_id,counts))
    chi2_features=chi2(features,labels==rule_id)
    indices=np.argsort(chi2_features[0])
    feature_names=np.array(tfidf.get_feature_names())[indices]
    unigrams=[v for v in feature_names if len(v.split())==1]
    bigrams=[v for v in feature_names if len(v.split())==2]
    print("# 'most corelated unigrams:'\n:{}".format('\n.'.join(unigrams[-5:])))
    print("# 'most corelated bigrams:'\n:{}".format('\n.'.join(bigrams[-5:])))
          
          #%%
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
X_train,X_test,y_train,y_test=train_test_split(df['X'],rule_id_df,random_state=0,test_size=0.20)
count_vect=CountVectorizer()
tfidf=TfidfTransformer(use_idf=True)
X_train_counts=count_vect.fit_transform(X_train)
X_train_tf_idf=tfidf.fit_transform(X_train_counts)


clf=LinearSVC(C=1,random_state=1)
clf.fit(X_train_tf_idf,y_train)
#%%
X_test_counts=count_vect.transform(X_test)
X_test_tf_idf=tfidf.transform(X_test_counts)
ypred=clf.predict(X_test_tf_idf)
#%%
print(len(ypred))
#%%
from sklearn.metrics import classification_report
print(classification_report(ypred,y_test))

#%%
clf.predict(tfidf.transform(count_vect.transform(['Organization_Name	Duetsche Bank	ICICI ltd	Infosys ltd'])))
#%%
clf.predict(tfidf.transform(count_vect.transform(['city	jaipur	pune	delhi'])))

     #%%
clf.predict(tfidf.transform(count_vect.transform(['email	er.anantjain@gmail.com	jain.infinity@gmail.com	jainriya516@gmail.com'])))     
     #%%
x='helloword'
print(x[-3:])          
    
    
    
    