#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd 
import numpy as np 


# In[104]:


df=pd.read_csv("C:/Users/vknsr/Downloads/wine.csv")
df.head()


# In[105]:


df1=df.pop("Wine")
df1.shape


# In[106]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(df,df1,test_size=0.2)


# In[107]:


from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier()
from sklearn.feature_selection import RFE


# In[149]:


rfe1=RFE(estimator=rfc,n_features_to_select=5,step=1,verbose=2)


# In[109]:


rfe1.fit(x_train,y_train)


# In[110]:


rfe1.support_


# In[111]:


x_newtrain=rfe1.transform(x_train)
x_newtest=rfe1.transform(x_test)


# In[112]:


x_newtrain.shape


# In[142]:


y_pred=rfc.fit(x_train,y_train).predict(x_test)
accuracy_score(y_test,y_pred)


# In[143]:


y_pred1


# In[144]:


rfe1.get_feature_names_out()


# In[145]:


from sklearn.metrics import accuracy_score


# In[147]:


y_pred1=rfc.fit(x_newtrain,y_train).predict(x_newtest)
accuracy_score(y_pred1,y_test)


# In[155]:


from sklearn.feature_selection import RFECV
rfecv=RFECV(estimator=rfc,min_features_to_select=5,step=1,n_jobs=-1,scoring="r2",cv=6,verbose=2)


# In[156]:


rfecv.fit(x_train,y_train)


# In[157]:


rfecv.support_


# In[166]:


y_train


# In[158]:


x_train1=rfecv.transform(x_train)


# In[163]:


x_test1=rfecv.transform(x_test)


# In[167]:


rfc2 = RandomForestClassifier()
rfc2.fit(x_train1,y_train)


# In[168]:


accuracy_score(y_test,rfc2.predict(x_test1))


# In[ ]:




