#Name : Vishnu Kumar S.R


#load the packages for loading the file
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#load the csv file using pandas as pd
df=pd.read_csv("C:/Users/vknsr/Downloads/Iris.csv")
df = df.set_index('Id')

df.head() # get the top 5 values from the data set

df.describe() #describe the data set and its data types

#split the data for training and testing
x=df.iloc[:,1:2]
y=df.iloc[:,4]


#load the train_test_split package from sklearn
from sklearn.model_selection import train_test_split 


#fit the values of x,y
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


#load the preprocessing package from sklearn and fit the model
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()  
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test) 


#from tree import decision tree classifier
from sklearn.tree import DecisionTreeClassifier


classifier= DecisionTreeClassifier(criterion='entropy', random_state=0)  

#fit the training data in the model
classifier.fit(x_train,y_train)


#calculate the prediction value
y_pred= classifier.predict(x_test) 


y_pred


from sklearn import tree


#plot the decision tree
print(tree.plot_tree(classifier,filled=True))


from sklearn.metrics import accuracy_score

#get the accuracy score using sklear.metrics
accuracy_score(y_test,y_pred)

#Output:
0.4



