import streamlit as st
import sys

from src.pipelines.predict_pipeline import PredictPipeline
from src.custom_data import CustomData
from src.exception import CustomException
from src.logger import logging

# streamlit app
st.title('Customer Churn Prediction')

if 'predicted' not in st.session_state:
    st.session_state['predicted'] = False

if not st.session_state['predicted']:
    st.session_state['predicted'] = True

    predict_pipeline = PredictPipeline(is_classifier=True)

    model, label_encoder_gender, one_hot_encoder_geo, scaler = predict_pipeline.get_objects()

    try:
        # User input
        geography = st.selectbox('Geography', one_hot_encoder_geo.categories_[0])
        gender = st.selectbox('Gender', label_encoder_gender.classes_)
        age = st.slider('Age', 18, 92)
        balance = st.number_input('Balance')
        credit_score = st.number_input('Credit Score', min_value=0)
        estimated_salary = st.number_input('Estimated Salary', min_value=0)
        tenure = st.slider('Tenure', 0, 10)
        num_of_products = st.slider('Number of Products', 1, 4)
        has_cr_card = st.selectbox('Has Credit Card', [0, 1])
        is_active_member = st.selectbox('Is Active Member', [0, 1])

        data=CustomData(credit_score= credit_score,
                        geography= geography,
                        gender= gender,
                        age= age,
                        tenure= tenure,
                        balance= balance,
                        number_of_products= num_of_products,
                        has_cr_card= has_cr_card,
                        is_active_member= is_active_member,
                        estimated_salary= estimated_salary,
                        exited= -1,
                        is_classifier= True)

        df = data.get_data_as_data_frame()

        logging.info("Data for Prediction:")
        logging.info(str(data))
        logging.info("Before Prediction")

        prediction = predict_pipeline.predict(df)
        prediction_probability = prediction[0][0]

        st.write(f'Churn Probability: {prediction_probability:.2f}')

        if prediction_probability > 0.5:
            st.write('The customer is likely to churn.')
        else:
            st.write('The customer is not likely to churn.')

        st.session_state['predicted'] = False

    except Exception as e:
        raise CustomException(e,sys)

