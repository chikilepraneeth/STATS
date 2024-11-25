#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[2]:


iris = sns.load_dataset('iris')


# In[6]:


print(iris.head(30))


# In[4]:


means=[]
for i in range(100):
    sample = iris.sample(n=30, random_state=np.random.randint(1, 1000))
    length = sample['sepal_length'].mean()
    means.append(length)


# In[5]:


plt.figure(figsize=(8, 5))
plt.hist(means, bins=15, color='blue', alpha=0.7, edgecolor='black')
plt.title('Histogram of Sample Means (Sepal Length)')
plt.xlabel('Mean Sepal Length')
plt.ylabel('Frequency')
plt.show()


# In[7]:


original_data, sample_data = train_test_split(iris, test_size=0.2, random_state=42)


# In[15]:


plt.figure(figsize=(10, 10))


# In[16]:


plt.subplot(1, 2, 1)
plt.scatter(original_data['sepal_length'], original_data['sepal_width'], color='blue', alpha=0.6, label='Original Data')
plt.title('Original Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[17]:


plt.subplot(1, 2, 2)
plt.scatter(sample_data['sepal_length'], sample_data['sepal_width'], color='green', alpha=0.6, label='Sample Data')
plt.title('Sampled Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[18]:


plt.subplot(1, 2, 1)
plt.scatter(original_data['sepal_length'], original_data['sepal_width'], color='blue', alpha=0.6, label='Original Data')
plt.title('Original Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.subplot(1, 2, 2)
plt.scatter(sample_data['sepal_length'], sample_data['sepal_width'], color='green', alpha=0.6, label='Sample Data')
plt.title('Sampled Data')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# Excersice 3
# 

# In[19]:


from scipy.stats import ttest_ind


# In[20]:


setosa = iris[iris['species'] == 'setosa']['petal_length']
versi = iris[iris['species'] == 'versicolor']['petal_length']


# In[23]:


t_test, p_value = ttest_ind(setosa, versi)


# In[25]:


print(f"Setosa Mean Petal Length: {setosa.mean():.2f}")
print(f"Versicolor Mean Petal Length: {versi.mean():.2f}")
print(f"T-statistic: {t_test:.2f}")
print(f"P-value: {p_value:.4f}")


# In[26]:


alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The means are significantly different.")
else:
    print("Fail to reject the null hypothesis: No significant difference in means.")


# In[36]:


from statsmodels.stats.weightstats import ztest


# In[37]:


setosa = iris[iris['species'] == 'setosa']['sepal_length']
z_stat, p_value = ztest(setosa, value=5.0)


# In[38]:


print(f"Z-Statistic: {z_stat:.2f}")
print(f"P-Value: {p_value:.4f}")


# In[39]:


alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The mean sepal length is significantly different from 5.0.")
else:
    print("Fail to reject the null hypothesis: The mean sepal length is not significantly different from 5.0.")


# In[40]:


from scipy.stats import f_oneway


# In[41]:


setosa = iris[iris['species'] == 'setosa']['petal_width']
versi = iris[iris['species'] == 'versicolor']['petal_width']
vir = iris[iris['species'] == 'virginica']['petal_width']


# In[42]:


f_stat, p_value = f_oneway(setosa, versi, vir)


# In[43]:


print(f"F-Statistic: {f_stat:.2f}")
print(f"P-Value: {p_value:.4f}")


# In[44]:


if p_value < alpha:
    print("Reject the null hypothesis: At least one mean is significantly different.")
else:
    print("Fail to reject the null hypothesis: All means are approximately equal.")


# In[45]:


correlation = iris['sepal_length'].corr(iris['petal_length'])

print(f"Correlation between Sepal Length and Petal Length: {correlation:.2f}")


# In[ ]:




