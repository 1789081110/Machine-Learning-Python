#!/usr/bin/env python
# coding: utf-8

# In[1]:


#load the important libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[117]:


df=pd.read_csv("C:/Users/vknsr/Downloads/Train.csv") #load the data set
df_test=pd.read_csv("C:/Users/vknsr/Downloads/Test.csv")
df.head(10)


# In[3]:


df.isnull().sum()


# In[4]:


df.drop(['society','availability','area_type','balcony'],axis=1,inplace=True)


# In[5]:


df.head()


# In[6]:


df.isnull().sum()


# In[7]:


df.dropna(inplace=True)


# In[52]:


df.info()


# In[12]:


df['size'].unique()


# In[13]:


df['bhk'] = df['size'].apply(lambda x: int(x.split(' ')[0]))


# In[15]:


df.head()


# In[16]:


df.drop(['size'],axis=1,inplace=True)


# In[17]:


df.head()


# In[18]:


df.total_sqft.unique()


# In[19]:


def to_float(x):
    try:
        return float(x)
    except:
        return False
    return True
        


# In[51]:


df['total_sqft'].apply(to_float)
df.head()


# In[32]:


def convert_to_num(x):
    lam=x.split('-')
    if(len(lam)==2):            
        return (float(lam[0])+float(lam[1]))/2
    try:
        return float(x)
    except:
        return None


# In[36]:


df1.info()


# In[57]:


df2 = df1.copy()


# In[58]:


df2['price_per_sqft'] = df2['price']*100000 / df2['total_sqft']
df2.head()


# In[61]:


df2['location'] = df2['location'].apply(lambda x: x.strip())
loc_stats=df2.location.value_counts()


# In[63]:


loc_less_than_10 = loc_stats[loc_stats<=10]
loc_less_than_10


# In[64]:


df2.location = df2.location.apply(lambda x: 'other' if x in loc_less_than_10 else x)
df2.head()


# In[65]:


len(df2.location.unique())


# In[89]:


df3 = df2[ ~(df2.total_sqft / df2.bhk < 300) ]
df3.shape
df3


# In[91]:


def remove_outlier_from_price_per_sqft(df):
    df_out = pd.DataFrame()
    for key,sub in df.groupby('location'):
        m = np.mean( sub.price_per_sqft )
        st = np.std( sub.price_per_sqft )
        reduce_df = sub[( sub.price_per_sqft>(m-st) ) & ( sub.price_per_sqft<=(m+st) ) ]
        df_out = pd.concat( [df_out, reduce_df],ignore_index=True )
    return df_out


# In[92]:


df4 = remove_outlier_from_price_per_sqft(df3)
df4.shape


# In[93]:


df4.describe()


# In[94]:


def remove_bhk_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        bhk_stats = {}
        for bhk, bhk_df in location_df.groupby('bhk'):
            bhk_stats[bhk] = {
                'mean': np.mean(bhk_df.price_per_sqft),
                'std': np.std(bhk_df.price_per_sqft),
                'count': bhk_df.shape[0]
            }
        for bhk, bhk_df in location_df.groupby('bhk'):
            stats = bhk_stats.get(bhk-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, bhk_df[bhk_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df5 = remove_bhk_outliers(df4)
df5.shape


# In[95]:


df6 = df5[~(df5.bath > df5.bhk+2)]
df6.head()


# In[96]:



df6.shape


# In[97]:


df7 = df6.drop(['price_per_sqft'],axis='columns')
df7.head()


# In[98]:


dummies = pd.get_dummies(df7.location)
dummies.head()


# In[99]:


df8 = pd.concat([df7,dummies.drop('other',axis='columns')],axis='columns')
df8.head()


# In[100]:


df8.drop('location',axis='columns',inplace=True)
df8.head()


# In[101]:


x = df8.drop('price',axis=1)
y = df8['price']


# In[102]:


x.shape


# In[103]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=101)


# In[120]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train,y_train)
y_pred=lr.score(X_test,y_test)


# In[134]:


submission=pd.DataFrame([{"Target":y_pred}])
submission.to_csv('submission',index=False)


# In[135]:


pd.read_csv('submission.csv').mean()


# In[ ]:




