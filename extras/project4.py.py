#!/usr/bin/env python
# coding: utf-8

# In[1]:

import sys
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[2]:

train_data  = sys.argv[1]
test_data = sys.argv[2]
train= pd.read_csv(train_data,encoding='ISO-8859-1')


# In[3]:


test= pd.read_csv(test_data,encoding='ISO-8859-1',header =0,names = ['v1','v2','v3','v4','v5'])


# In[4]:


test = test[['v1','v2']]


# In[5]:


train = train[['v1','v2']]


# In[6]:



feature_extraction = TfidfVectorizer()
X = feature_extraction.fit_transform(train['v2'].values)
orig_vocab = feature_extraction.get_feature_names()


# In[7]:


X_train, X_test, y_train, y_test = train_test_split(X, train['v1'], test_size=0.33, random_state=42)


# In[8]:


clf = SVC(probability=True, kernel='rbf',gamma = 'scale')
clf.fit(X_train, y_train)


# In[11]:


feature_extraction = TfidfVectorizer(vocabulary=orig_vocab)
Y = feature_extraction.fit_transform(test['v2'].values)


# In[12]:


prd = clf.predict(Y)


# In[14]:


acc = accuracy_score(test['v1'], prd, normalize=True)


# In[15]:


for i in range(len(prd)):
    print(i, prd[i])
print("Accuracy:",acc*100)


# In[ ]:
