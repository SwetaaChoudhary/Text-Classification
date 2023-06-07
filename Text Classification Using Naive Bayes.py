#!/usr/bin/env python
# coding: utf-8

# In[3]:


import nltk
import spacy
import pandas as pd


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[6]:


#df=pd.read_csv(r"C:\Users\HP\Downloads\archive (2)\raw_data.csv")
#df


# In[7]:


from sklearn.datasets import fetch_20newsgroups
data = fetch_20newsgroups()
data.target_names


# In[8]:


data


# In[9]:


categories =['alt.atheism',
 'comp.graphics',
 'comp.os.ms-windows.misc',
 'comp.sys.ibm.pc.hardware',
 'comp.sys.mac.hardware',
 'comp.windows.x',
 'misc.forsale',
 'rec.autos',
 'rec.motorcycles',
 'rec.sport.baseball',
 'rec.sport.hockey',
 'sci.crypt',
 'sci.electronics',
 'sci.med',
 'sci.space',
 'soc.religion.christian',
 'talk.politics.guns',
 'talk.politics.mideast',
 'talk.politics.misc',
 'talk.religion.misc']


# In[10]:


train = fetch_20newsgroups(subset="train",categories=categories)


# In[11]:


test= fetch_20newsgroups(subset="test",categories=categories)


# In[12]:


print(train.data[5])


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline


# In[14]:


model= make_pipeline(TfidfVectorizer(),MultinomialNB())


# In[15]:


model.fit(train.data,train.target)


# In[16]:


labels = model.predict(test.data)


# In[20]:


from sklearn.metrics import confusion_matrix
mat = confusion_matrix(test.target,labels)
sns.heatmap(mat.T,square=True,annot =True, fmt="d",cbar= False,xticklabels =train.target_names,yticklabels=test.target_names)
plt.xlabel("true label")
plt.ylabel("predicted label")


# In[21]:


def predict_category(s,train=train,model=model):
    pred = model.predict([s])
    return train.target_names[pred[0]]

