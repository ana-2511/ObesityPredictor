import streamlit as st
import pandas as pd
import joblib
from io import StringIO
import base64
import xgboost
from sklearn import preprocessing, model_selection, ensemble  # Adjust imports as per your specific usage


# Set the page configuration
st.set_page_config(page_title='Obesity Predictor - Your Friendly App', layout='wide')

# Inject custom CSS for background image and styling
st.markdown(
    f"""
    <style>
    .main {{
        background-image: url('https://www.baptisthealth.com/-/media/images/migrated/blog-images/teaser-images/gettyimages-1214458445-1280x853.jpg?rev=be733d68f1514cefa49b26a225626fae');
        background-size: cover;
        padding: 20px;
    }}
    .stButton>button {{
        background-color: #4CAF50;
        color: white;
    }}
    .stTextInput>div>div>input {{
        background-color: #f0f2f6;
        color: black
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Load the trained model and scaler
model = joblib.load('xgb_new_model.pkl.gz')
scaler = joblib.load('scaler_retrained.pkl')

# Map for obesity levels
obesity_levels = {
    0: 'Under_Weight',
    1: 'Normal_Weight',
    2: 'Overweight_Level_I',
    3: 'Overweight_Level_II',
    4: 'Obesity_Type_I',
    5: 'Obesity_Type_II',
    6: 'Obesity_Type_III'
}

# Function to calculate BMI
def calculate_bmi(weight, height):
    return weight / (height ** 2)

# Function to determine the normal weight range for a given height
def normal_weight_range(height):
    min_normal_bmi = 18.5
    max_normal_bmi = 24.9
    min_normal_weight = min_normal_bmi * (height ** 2)
    max_normal_weight = max_normal_bmi * (height ** 2)
    return min_normal_weight, max_normal_weight

# Function to predict obesity level
def predict_obesity(gender, age, height, weight, family_history, high_caloric_food, alcohol_consumption, transportation_mode, smoking):
    # Preprocess inputs
    gender = 0 if gender == 'Male' else 1
    family_history = 1 if family_history == 'yes' else 0
    high_caloric_food = 1 if high_caloric_food == 'yes' else 0
    smoking = 1 if smoking == 'Yes' else 0

    transportation_map = {'Automobile': 0, 'Motorbike': 1, 'Bike': 2, 'Public_Transportation': 3, 'Walking': 4}
    alcohol_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}

    transportation_mode = transportation_map[transportation_mode]
    alcohol_consumption = alcohol_map[alcohol_consumption]

    # Create input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height': [height],
        'Weight': [weight],
        'family_history_with_overweight': [family_history],
        'FAVC': [high_caloric_food],
        'CALC': [alcohol_consumption],
        'MTRANS': [transportation_mode],
        'SMOKE': [smoking]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Make predictions
    prediction = model.predict(input_data_scaled)
    predicted_class = obesity_levels[prediction[0]]

    return predicted_class

# Function to generate a weight management report
def generate_report(username, gender, age, height, weight, family_history, high_caloric_food, alcohol_consumption, transportation_mode, smoking, predicted_level, bmi, min_normal_weight, max_normal_weight):
    report = StringIO()
    report.write(f"Weight Management Report for {username}\n")
    report.write(f"Gender: {gender}\n")
    report.write(f"Age: {age}\n")
    report.write(f"Height: {height:.2f} meters\n")
    report.write(f"Weight: {weight:.2f} kg\n")
    report.write(f"Family History with Overweight: {family_history}\n")
    report.write(f"Frequently Consume High Caloric Food: {high_caloric_food}\n")
    report.write(f"Consumption of Alcohol: {alcohol_consumption}\n")
    report.write(f"Transportation Mode: {transportation_mode}\n")
    report.write(f"Smoking Rate: {smoking}\n\n")
    report.write(f"Predicted Obesity Level: {predicted_level}\n")
    report.write(f"Your BMI: {bmi:.2f}\n\n")
    
    if predicted_level != 'Normal_Weight':
        report.write(f"Based on your height, your normal weight range should be between {min_normal_weight:.2f} kg and {max_normal_weight:.2f} kg.\n")
        report.write("Here are some suggestions for managing your weight:\n")
        report.write("- Follow a balanced diet with appropriate caloric intake.\n")
        report.write("- Engage in regular physical activity.\n")
        report.write("- Avoid high-caloric food and excessive alcohol consumption.\n")
        report.write("- Consult a healthcare professional for personalized advice.\n")
    
    return report.getvalue()

# Function to render the obesity prediction form
def obesity_prediction_form(username):
    st.header('Enter Details')
    
    if 'age' not in st.session_state:
        st.session_state.age = 14
    if 'height' not in st.session_state:
        st.session_state.height = 1.0
    if 'weight' not in st.session_state:
        st.session_state.weight = 20.0

    gender = st.selectbox('Select Gender', ['Male', 'Female'])
    age = st.number_input('Enter Age', min_value=14, max_value=61, step=1, value=st.session_state.age)
    height = st.number_input('Enter Height (in meters)', min_value=1.0, max_value=2.5, step=0.01, value=st.session_state.height)
    weight = st.number_input('Enter Weight (in kg)', min_value=20.0, max_value=200.0, step=0.1, value=st.session_state.weight)
    family_history = st.selectbox('Family History with Overweight', ['yes', 'no'])
    high_caloric_food = st.selectbox('Do you frequently consume high caloric food?', ['yes', 'no'])
    alcohol_consumption = st.selectbox('Consumption of alcohol', ['no', 'Sometimes', 'Frequently', 'Always'])
    transportation_mode = st.selectbox('Transportation mode', ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking'])
    smoking = st.selectbox('Smoking Rate', ['Yes', 'No'])

    if st.button('Predict Obesity Level'):
        # Store the input values in session state
        st.session_state.age = age
        st.session_state.height = height
        st.session_state.weight = weight

        # Call prediction function
        predicted_level = predict_obesity(gender, age, height, weight, family_history, high_caloric_food, alcohol_consumption, transportation_mode, smoking)
        bmi = calculate_bmi(weight, height)
        min_normal_weight, max_normal_weight = normal_weight_range(height)
        
        st.subheader(f'Predicted Obesity Level: {predicted_level}')
        st.subheader(f'Your BMI: {bmi:.2f}')
        
        if predicted_level != 'Normal_Weight':
            st.subheader('Weight Management Report')
            st.write(f'Based on your height, your normal weight range should be between {min_normal_weight:.2f} kg and {max_normal_weight:.2f} kg.')
            st.write('Here are some suggestions for managing your weight:')
            st.write('- Follow a balanced diet with appropriate caloric intake.')
            st.write('- Engage in regular physical activity.')
            st.write('- Avoid high-caloric food and excessive alcohol consumption.')
            st.write('- Consult a healthcare professional for personalized advice.')

            report_content = generate_report(username, gender, age, height, weight, family_history, high_caloric_food, alcohol_consumption, transportation_mode, smoking, predicted_level, bmi, min_normal_weight, max_normal_weight)
            
            st.download_button(label='Download Report', data=report_content, file_name='weight_management_report.txt', mime='text/plain')


# Function to render the login form
def login_form():
    st.title('Obesity Predictor - Your Friendly App')
    st.subheader('Login to Obesity Prediction App')
    username = st.text_input('Username', key='login_username')
    if st.button('Login'):
        if username:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()  # Rerun to update the UI

# Function to render the logout button
def logout_button():
    if st.button('Logout'):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.experimental_rerun()

# Main function to run the app
def main():
    # Check if user is logged in
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    # Render login page if not logged in
    if not st.session_state.logged_in:
        login_form()
    else:
        st.title('Obesity Predictor - Your Friendly App')
        st.header(f'Welcome, {st.session_state.username}!')
        logout_button()
        obesity_prediction_form(st.session_state.username)

if __name__ == '__main__':
    main()
