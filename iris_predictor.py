import streamlit as st 
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#loading the dataset
iris = datasets.load_iris()

#choosing the model
clf = RandomForestClassifier(n_estimators = 100)
clf.fit(iris.data, iris.target)
pred = clf.predict(iris.data)

#splitting, training and showing the accuracy of the model
scores = cross_val_score(clf, iris.data, iris.target, cv= 10)
print('Accuracy:', scores)
print('Average Accuracy:', round(sum(scores)/10))
#The accuracy of the model is 1, therefore it is very good at predicting the classes

#STREAMLIT INTERFACE
#Title

st.title('DATA SCIENCE STREAMLIT PROJECT')
#Header 


st.header('IRIS DATASET PREDICTOR')
#Making a dataset for the features
dataset = pd.DataFrame(iris.data, columns= iris.feature_names)

#Seperating the features and finding the minimum, maximum and mean for the user to choose the values

#Sepal length

x = dataset['sepal length (cm)']
min_x = np.min(x)
max_x = np.max(x)
mean_x = round(np.mean(x))
mean_x = float(mean_x)
level = st.slider("Select the sepal length value", min_x, mean_x, max_x)
st.text('Sepal length: {}'.format(level))
st.text('')

#Sepal width
w = dataset['sepal width (cm)']
min_w = np.min(w)
max_w = np.max(w)
mean_w = (round(np.mean(w)))
mean_w = float(mean_w)
level0 = st.slider("Select the sepal width value", min_w, mean_w, max_w)
st.text('Sepal width: {}'.format(level0))
st.text('')

#Petal length
y = dataset['petal length (cm)']
min_y = np.min(y)
max_y = np.max(y)
mean_y = round(np.mean(y))
mean_y = float(mean_y)
level2 = st.slider("Select the petal length value", min_y, mean_y, max_y)
st.text('Petal length: {}'.format(level2))
st.text('')

#Petal width
z = dataset['petal width (cm)']
min_z = np.min(z)
max_z = np.max(z)
mean_z = (round(np.mean(z)))
mean_z = float(mean_z)
level3 = st.slider("Select the petal width value", min_z, mean_z, max_z)
st.text('Petal width: {}'.format(level3))
st.text('')


#Storing the inputs and class prediction
input_data = pd.DataFrame([level,level0, level2, level3])
input_data = np.array(input_data).reshape(1,4)
pred2 = clf.predict(input_data)
p = 'name'
if pred2 == 0:
    p = 'setosa'
if pred2 == 1:
    p = 'versicolor'
if pred2 == 2:
    p == 'virginica'
 
#Creating a button that shows the class it predicted
if st.button('Predict'):
    st.write(f'The class of the flower is {p}')

