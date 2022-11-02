import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("C:/Users/vknsr/OneDrive/Documents/ML/Iris.csv")
df
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

df["Species"]=le.fit_transform(df["Species"])
df["Species"]

df1=df.pop("Species")
df1.head()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df,df1,test_size=0.2)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
from sklearn.metrics import accuracy_score
accuracy_score(y_pred,y_test)
