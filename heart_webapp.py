import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title='Heart_test')
st.title("Disease Classification")

@st.cache(allow_output_mutation=True)
def get_model():
    return joblib.load('heart_knn_model.joblib')

cp=st.text_input("Enter cp:","")
thalach=st.text_input("Enter thalach:","")
exang=st.text_input("Enter exang:","")
oldpeak=st.text_input("Enter oldpeak:","")

if st.button("Check Disease"):
    values=[cp,thalach,exang,oldpeak]
    num_values=[]
    for x in values:
        num_values.append(float(x))
    
    #2 dimension
    num_values=np.asarray(num_values).reshape(1,-1)
    predictions=get_model().predict(num_values)
    predictions=int(predictions)
    happy="happyface.png"
    care="careface.png"
    if predictions==0:
        st.write("Negative")
        st.image(happy,caption="No Disease")
    elif predictions==1:
        st.write("positive")
        st.image(care,caption="Take Care")
   