# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import joblib

def main():
    html_temp = """
    <div style="background-color:lightpink;padding:16px">
    <h2 style= "color:black";text-align:center> Health Insurance Cost Prediction Web App
    </div>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    
    model= joblib.load('model_joblib_gb')
    p1= st.slider('Enter Your Age',18,100)
    s1= st.selectbox('Sex',('Male','Female'))
    
    if s1=='Male':
        p2=1
    else:
        p2=0
    
    p3= st.number_input("Enter your BMI Value")
    
    p4=st.slider('Enter the No. of Children',0,5)
    s2=st.selectbox('Smoker',('yes','no'))
    
    if s2=='yes':
        p5=1
    else:
        p5=0
    
    p6= st.slider('Enter your region',1,4)
    
    if st.button('Predict'):
        pred=model.predict([[p1,p2,p3,p4,p5,p6]])
        
        st.success('Your Insurance Cost is {}'.format(round(pred[0],2)))
    
    
    
    
    
    
 
    
    
if __name__ == '__main__' :
    main()