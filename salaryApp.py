# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 19:26:00 2020

@author: User
"""
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import seaborn as sns
from sklearn.linear_model import LinearRegression

data = pd.read_csv("pay.csv")
X = np.array(data["Years of Experience"]).reshape(-1,1)
Lin_reg = LinearRegression()
Lin_reg.fit(X,np.array(data["Salary"]))


st.title("SALARY PREDICTOR")
navigation = st.sidebar.radio("Navigation",["Home","Prediction","Contribute"])

if navigation == "Home":
    st.image("salary.jpg",width=350)
    if st.checkbox("Show Data"):
        st.table(data)
        
    graph = st.selectbox("Select the kind of plot", ["Non-Interactive","Interactive"])
    val = st.slider("Filter data by years of Experience",0,15)
    data = data.loc[data["Years of Experience"]>=val]
    if graph == "Non-Interactive":
        plt.figure(figsize= (10,5))
        plt.scatter(data["Years of Experience"], data["Salary"])
        plt.ylim(0)
        plt.xlabel("Years of Experience")
        plt.ylabel("Salary")
        plt.tight_layout()
        st.pyplot()
    
    if graph == "Interactive":
        layout =go.Layout(
           xaxis= dict(range=[0,16]),
           yaxis= dict(range=[0,210000])
        
        ) 
        fig = go.Figure(data=go.Scatter(x=data["Years of Experience"], y=data["Salary"], mode='markers'),
        layout=layout)
        st.plotly_chart(fig)
                
                
      
        
if navigation == "Prediction":
    st.image("salary.jpg",width=350)
    st.header("Predict your Salary")
    val = st.number_input("Enter your years of Experience",0.00,10.00, step=0.20)
    val = np.array(val).reshape(1,-1)
    prediction = Lin_reg.predict(val)[0]
    
    if st.button("Click to Predict"):
        st.success(f"Your predicted salary is {round(prediction)}")
    
    
if navigation == "Contribute":
    st.image("salary.jpg",width=350)
    st.header("Contribute to this dataset")
    Experience = st.number_input("Enter your years of Experience",0.0,15.0)
    Salary = st.number_input("Enter your salary",0.00,1000000.00, step=1000.0)
    if st.button("Click to Submit"):
        to_add = {"Years of Experience":Experience,"Salary":Salary}
        to_add = pd.DataFrame(to_add,index=[1])
        to_add.to_csv("C:\\Users\\User\\Documents\\pay.csv",header=False,index=False)
        st.success("Submitted")
   
    
    
    
    
    
