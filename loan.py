import streamlit as st
import pickle
import numpy as np
import os

# Define the model file path
MODEL_PATH = os.path.join("C:\\Users\\Prasad\\ML Project\\Loan Prediction", "train_model.pkl")

# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as pkl:
        train_model = pickle.load(pkl)
    st.success("Model loaded successfully!")
except FileNotFoundError:
    st.error(f"Error: Model file not found at {MODEL_PATH}. Please check the file path.")
    train_model = None
except pickle.UnpicklingError:
    st.error("Error: Failed to load the model. The file may be corrupted.")
    train_model = None
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
    train_model = None

def main():
    st.title("Loan Eligibility Prediction")
    st.markdown("""Use this app to predict loan eligibility based on applicant details.""")

    # Input fields
    left, right = st.columns((2,2))
    gender = left.selectbox('Gender', ('Male', 'Female'))
    married = right.selectbox('Married', ('Yes', 'No'))
    dependent = left.selectbox('Dependents', ('None', 'One', 'Two', 'Three'))
    education = right.selectbox('Education', ('Graduate', 'Not Graduate'))
    self_employed = left.selectbox('Self-Employed', ('Yes', 'No'))
    applicant_income = right.number_input('Applicant Income', min_value=0.0, step=500.0)
    coApplicantIncome = left.number_input('Coapplicant Income', min_value=0.0, step=500.0)
    loanAmount = right.number_input('Loan Amount', min_value=0.0, step=500.0)
    loan_amount_term = left.number_input('Loan Tenor (in months)', min_value=1, step=1)
    creditHistory = right.number_input('Credit History', min_value=0.0, max_value=1.0, step=1.0)
    propertyArea = st.selectbox('Property Area', ('Semiurban', 'Urban', 'Rural'))

    if st.button('Predict'):
        if train_model is not None:
            result = predict(gender, married, dependent, education, self_employed,
                            applicant_income, coApplicantIncome, loanAmount,
                            loan_amount_term, creditHistory, propertyArea)
            st.success(f"You are {result} for the loan.")
        else:
            st.error("Model is not loaded. Please check for errors.")

def predict(gender, married, dependent, education, self_employed, applicant_income,
            coApplicantIncome, loanAmount, loan_amount_term, creditHistory, propertyArea):
    # Encoding categorical variables
    gen = 0 if gender == 'Male' else 1
    mar = 0 if married == 'Yes' else 1
    dep = {'None': 0, 'One': 1, 'Two': 2, 'Three': 3}[dependent]
    edu = 0 if education == 'Graduate' else 1
    sem = 0 if self_employed == 'Yes' else 1
    pro = {'Semiurban': 0, 'Urban': 1, 'Rural': 2}[propertyArea]
    loAm = loanAmount / 1000
    cap = coApplicantIncome / 1000

    # Creating input array
    input_features = np.array([[gen, mar, dep, edu, sem, applicant_income, cap, loAm, loan_amount_term, creditHistory, pro]])

    # Debugging: Print input shape
    print(f"Input shape: {input_features.shape}")
    print(f"Model expects {train_model.n_features_in_} features.")

    if train_model is not None:
        try:
            prediction = train_model.predict(input_features)
            print(f"Prediction output: {prediction}")
            return 'Not Eligible' if prediction[0] == 0 else 'Eligible'
        except Exception as e:
            print(f"Prediction Error: {e}")
            return "Prediction Error"
    else:
        return "Model not loaded"

if __name__ == '__main__':
    main()
