

from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


# In[2]:


df_train=pd.read_csv("C:/Users/vknsr/Downloads/Dataset_Student_Academic_Success_Prediction/train.csv",nrows=10000)
df_test=pd.read_csv("C:/Users/vknsr/Downloads/Dataset_Student_Academic_Success_Prediction/test.csv")
df_train.head()


# In[3]:


df_train.head()


# In[4]:


from sklearn.naive_bayes import GaussianNB


# In[7]:


ga=GaussianNB()
df1=df_train.pop('Target')


# In[9]:


ga.fit(df_train,df1)


# In[10]:


y_pred=ga.predict(df_test)


# In[12]:


y_pred.mean()


# In[ ]:




