import streamlit as st
import pickle
import pandas as pd
import numpy as np

# load model 
XGboost_model = pickle.load(open('Model_XGBoost.sav','rb'))

# create title 
st.title('Late Product Prediction')

# create form input
# seperate into 2 columns

left,right = st.columns(2)

with left:
    warehouse_block = st.number_input ('input warehouse block')

with right:
    shipment_model = st.number_input('input shipment mode')

with left:
    customer_calls = st.number_input('input how many customer calls')

with right:
    customer_rating = st.number_input('input customer rating')

with left:
    product_cost = st.number_input('input product cost')

with right:
    product_priority = st.number_input('input product priority')
    
with left:
    product_importance = st.number_input('input product importance')

with right:
    gender = st.number_input('input gender')

with left:
    discount_offerd = st.number_input('input discount offered')

with right:
    product_weight = st.number_input('input product weight')


# prediction code
prediction_result = ''

# prediction button
st.button('Product on time prediction')

columns_predict = ['warehouse_block','mode_of_shipment','customer_care_calls','customer_rating', 'cost_of_the_product',
                   'prior_purchases','product_importance','gender','discount_offered','weight_in_gms']

input_user = [[warehouse_block,shipment_model,customer_calls,customer_rating,product_cost,product_priority,product_importance,gender,discount_offerd,product_weight]]
input_user = pd.DataFrame(input_user, columns=columns_predict)

product_prediction = XGboost_model.predict(input_user)

if (product_prediction[0] == 1):
    prediction_result = 'Your product prediction late'
else:
    prediction_result = 'Your product prediction on-time'
st.success(prediction_result)