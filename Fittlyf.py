#!/usr/bin/env python
# coding: utf-8

# In[54]:


#Loading the libraries to read the data set
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[55]:


# using pandas read the csv file(dataset)
df=pd.read_csv("C:/Users/vknsr/Downloads/Fytlyff_DS_Interview.csv")
pd.DataFrame(df) # coverting all values to dataframe


# In[56]:


#reading the first five data sets of the entire data set, data set contains both catogorical and numeric values and null values
df.head()


# In[57]:


# getting the column names used from the data set
df.columns


# In[58]:


#getting the total number of null values column wise
df.isnull().sum()


# In[59]:


# knowing more info about the data set before cleaning the data
df.describe()


# In[60]:


#data cleaning is performed using replace function
def data_cleaning():
    df=df.replace(to_replace=np.NaN,value=0)
    df['Month'] = pd.to_datetime(df['Month'],format='%b').dt.month
    df=df.replace(to_replace="Came_From_Google",value="Google")
    df=df.replace(to_replace="Landed_on_the_page_Directly",value="Direct_traffic")


# In[61]:


#descriptive status
def descriptive_status():
    df.describe()
    df.info()


# In[62]:


a=df.groupby('Which_Place_in_India?') # to get the city max for each city we use group by method
b=a.max(5)
b


# In[63]:


# dividing both the columns of b and keep it in c
c=b['How_many_Landed_on_our_Page?']/b['How_many_Landed_on_the_our_Page_and_clicked_on_a_button_and_started_filling_the_Form_and_Completed_and_submited_the_form?']


# In[64]:


c  # from this we can see chennai is highest


# In[65]:


d=df["Year"]=2021 # assigning all values of 2021 to d


# In[66]:


from sklearn.metrics import mean_absolute_percentage_error #imported the MAPE from sklearn
def pred_future():
    b['How_many_Landed_on_the_our_Page_and_clicked_on_a_button_and_started_filling_the_Form_and_Completed_and_submited_the_form?'].predict("Year"==2022)
    mean_absolute_percentage_error(df,d)


# In[72]:


# a line plot b/w year and filled and submitted the form
x=df["Year"]
y=df["How_many_Landed_on_the_our_Page_and_clicked_on_a_button_and_started_filling_the_Form_and_Completed_and_submited_the_form?"]
plt.plot(x,y,color="g",marker=",")
plt.ylabel("How_many_Landed_on_the_our_Page_and_clicked_on_a_button_and_started_filling_the_Form_and_Completed_and_submited_the_form")
plt.xlabel("Year")


# In[71]:


# visualization of the graph for moths b/w submitted and filled the form
import seaborn as sns
plt.figure(figsize=(15,4))
sns.set_style("darkgrid")
sns.lineplot(data=df, x="Month",
y="How_many_Landed_on_the_our_Page_and_clicked_on_a_button_and_started_filling_the_Form_and_Completed_and_submited_the_form?",
 ci=None, color="green", marker='o')


#                                          Thankyou!
