
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[236]:


df_train=pd.read_csv("C:/Users/vknsr/Downloads/Dataset_Student_Academic_Success_Prediction/train.csv")
df_test=pd.read_csv("C:/Users/vknsr/Downloads/Dataset_Student_Academic_Success_Prediction/test.csv")
df_test.head()


# In[237]:


from xgboost import XGBClassifier
xgb=XGBClassifier()


# In[238]:


get_ipython().system('pip install xgboost')


# In[239]:


df1=df_train.pop('Target')


# In[240]:


xg.fit(df_train_final, df1)


# In[241]:


y_pred=xg.predict(df_test)
y_pred


# In[182]:


submission=pd.DataFrame({"Target":y_ptest})
submission.to_csv('submission',index=False)


# In[183]:


pd.read_csv('submission.csv').mean()


# In[ ]:




