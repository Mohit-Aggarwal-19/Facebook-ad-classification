import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import sklearn.preprocessing as sk
import sklearn.model_selection as model_select
import sklearn.linear_model as lr_model
import sklearn.metrics as metrics

##import the data set and changing the encoding the to latin
training_set=pd.read_csv('Facebook_Ads_2.csv',encoding="latin1")
print(training_set.head())

##data Visualization
clicked=training_set[training_set['Clicked']==1]
not_clicked=training_set[training_set['Clicked']==0]
print("Total number of clicked records:",len(clicked),"Total number of clicked records:",len(not_clicked))

##Data analysis on seaborne

sns.scatterplot(training_set['Time Spent on Site'],training_set['Salary'],hue=training_set['Clicked'])
training_set['Salary'].hist(bins=40)
training_set['Time Spent on Site'].hist(bins=40)

##data cleansing
training_set.drop(['Names','emails','Country'],axis=1,inplace=True)
print(training_set.head())

##seperating the class column (labels) and input(features)

X=training_set.drop('Clicked',axis=1).values
Y=training_set['Clicked'].values
print(X[0:5])
print(Y[0:5])

##feature Scaling
X=sk.StandardScaler().fit_transform(X)
print(X[0:5])

##Model training
X_train,X_test,Y_train,Y_test=model_select.train_test_split(X,Y,test_size=0.2,random_state=42)
print(X_train.shape)
print(X_test.shape)


classifier=lr_model.LogisticRegression(random_state=42)
classifier.fit(X_train,Y_train)

##Model testing

Y_train_predict=classifier.predict(X_train)
print(Y_train_predict[0:50])
print(Y_train[0:50])
Y_test_predict=classifier.predict(X_test)
print(Y_test_predict[0:20])
print(Y_test[0:20])

##confusion matrix
cm_train=metrics.confusion_matrix(Y_train,Y_train_predict)
##sns.heatmap(cm_train,annot=True,fmt='d')
cm_test=metrics.confusion_matrix(Y_test,Y_test_predict)
sns.heatmap(cm_test,annot=True,fmt='d')

##final report
print(metrics.classification_report(Y_test,Y_test_predict))