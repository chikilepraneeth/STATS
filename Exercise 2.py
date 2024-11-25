#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[32]:


data2 = pd.read_csv('gender_submission.csv')


# In[33]:


data2.info()


# In[34]:


data2.head(10)


# In[35]:


data3=pd.read_csv('train.csv')


# In[36]:


data3.head()


# In[37]:


print(data3.columns)


# In[38]:


data3.sample(5)


# In[41]:


sns.barplot(x="Sex", y="Survived", data=data3)

print("Percentage of females who survived:", data3["Survived"][data3["Sex"] == 'female'].value_counts(normalize = True)[1]*100)

print("Percentage of males who survived:", data3["Survived"][data3["Sex"] == 'male'].value_counts(normalize = True)[1]*100)


# In[42]:


sns.barplot(x="Pclass", y="Survived", data=data3)

print("Percentage of Pclass = 1 who survived:", data3["Survived"][data3["Pclass"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 2 who survived:", data3["Survived"][data3["Pclass"] == 2].value_counts(normalize = True)[1]*100)

print("Percentage of Pclass = 3 who survived:", data3["Survived"][data3["Pclass"] == 3].value_counts(normalize = True)[1]*100)


# In[43]:


sns.barplot(x="SibSp", y="Survived", data=data3)

print("Percentage of SibSp = 0 who survived:", data3["Survived"][data3["SibSp"] == 0].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 1 who survived:", data3["Survived"][data3["SibSp"] == 1].value_counts(normalize = True)[1]*100)

print("Percentage of SibSp = 2 who survived:", data3["Survived"][data3["SibSp"] == 2].value_counts(normalize = True)[1]*100)


# In[45]:


sns.barplot(x="Parch", y="Survived", data=data3)
plt.show()


# In[46]:


data3["Age"] = data3["Age"].fillna(-0.5)

bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
data3['AgeGroup'] = pd.cut(data3["Age"], bins, labels = labels)



sns.barplot(x="AgeGroup", y="Survived", data=data3)
plt.show()


# In[ ]:




