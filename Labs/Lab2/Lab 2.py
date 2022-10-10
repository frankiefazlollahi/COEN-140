#!/usr/bin/env python
# coding: utf-8

# # Lab 2

# ## Part 1: Load Data

#     --Step 1: Reading data from four text files using NumPy and Pandas

# In[1]:


import pandas as pd
import numpy as np

df1 = pd.read_csv('anger-ratings.txt', header = None, sep = '\t')
df2 = pd.read_csv('fear-ratings.txt', header = None, sep = '\t')
df3 = pd.read_csv('joy-ratings.txt', header = None, sep = '\t')
df4 = pd.read_csv('sadness-ratings.txt', header = None, sep = '\t')

# df1


#     --Step 2: Combine four sub datasets

# In[2]:


dfs = [df1, df2, df3, df4]
df = pd.concat(dfs)
df.columns = ["ID", "Sentence", "Feeling", "Rating"]
df


#     --Step 3: Shuffle the new dataset

# In[3]:


df = df.sample(frac = 1)
df


#     --Step 4: Show the shape of the dataset

# In[4]:


shape = df.shape
print('Dataset shape: ', shape)


# ## Part 2: Plot word frequency for Top 30 words

#     --Step 1: Check missing value

# In[5]:


df.isnull().sum()


#     --Step 2: Check duplicate value

# In[6]:


dups = df.duplicated()
dups


#     --Step 3: Create term-document matrix

# In[7]:


import sklearn
from sklearn.feature_extraction.text import CountVectorizer


count_vect = CountVectorizer()
sparse_matrix = count_vect.fit_transform(df.Sentence)
#print(sparse_matrix)

#sparse_matrix.shape
#count_vect.vocabulary_

dataf = pd.DataFrame(sparse_matrix.todense())
dataf


#     --Step 4: Count frequency and sort them

# In[8]:


dataf.columns = count_vect.get_feature_names()
dataf = dataf.T
dataf


#     --Step 5: Plot the top 30 words frequencies

# In[9]:


dataf["Count"] = dataf.sum(axis=1)


# In[10]:


dataf


# In[12]:


dataf.sort_values("Count", ascending = False)
dataf


# In[28]:


import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

figure(figsize=(15, 10))

top30 = dataf.head(30)
#print(top30)
plt.bar(top30.index, top30['Count'])
plt.show()


# In[ ]:




