import streamlit as st
import pandas as pd
import pickle

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")

load_clf = pickle.load(open('Random_forest_model.pkl', 'rb'))

def user_input_features():
    st.sidebar.title('User Input Features')
    age = st.sidebar.number_input('Age', 29, 71, 40)

    sex_mapping = {'0-Female': 0, '1-Male': 1}
    sex = st.sidebar.selectbox('Sex', ['0-Female', '1-Male'])
    sex = sex_mapping[sex]

    cp_mapping = {'0-Typical Angina': 0, '1-Atypical Angina': 1, '2-Non-Anginal Pain': 2, '3-Asymptomatic': 3}
    cp = st.sidebar.selectbox('Chest Pain Type', ['0-Typical Angina', '1-Atypical Angina', '2-Non-Anginal Pain', '3-Asymptomatic'])
    cp = cp_mapping[cp]

    tres = st.sidebar.number_input('Resting Blood Pressure (mm Hg)', 94, 200, 120)
    
    chol = st.sidebar.number_input('Serum Cholesterol (mg/dl)', 126, 564, 200)

    fbs_mapping = {'0-False': 0, '1-True': 1}
    fbs = st.sidebar.selectbox('Fasting Blood Sugar', ['0-False', '1-True'])
    fbs = fbs_mapping[fbs]

    res_mapping = {'0-Normal': 0, '1-ST-T Wave Abnormality': 1, '2-Left Ventricular Hypertrophy': 2}
    res = st.sidebar.selectbox('Resting ECG Result', ['0-Normal', '1-ST-T Wave Abnormality', '2-Left Ventricular Hypertrophy'])
    res = res_mapping[res]

    tha = st.sidebar.number_input('Max Heart Rate Achieved', 71, 202, 150)

    exa_mapping = {'0-No': 0, '1-Yes': 1}
    exa = st.sidebar.selectbox('Exercise Induced Angina', ['0-No', '1-Yes'])
    exa = exa_mapping[exa]

    old = st.sidebar.number_input('Oldpeak', 0.0, 6.2, 2.0)

    slope_mapping = {'0-Upsloping': 0, '1-Flat': 1, '2-Downsloping': 2}
    slope = st.sidebar.selectbox('Slope of Peak Exercise ST Segment', ['0-Upsloping', '1-Flat', '2-Downsloping'])
    slope = slope_mapping[slope]

    ca = st.sidebar.slider('Number of Major Vessels', 0, 3, 1)

    thal_mapping = {'1-Normal': 1, '2-Fixed Defect': 2, '3-Reversible Defect': 3}
    thal = st.sidebar.selectbox('Thal Rate', ['1-Normal', '2-Fixed Defect', '3-Reversible Defect'])
    thal = thal_mapping[thal]

    data = {'age': age,
            'sex': sex, 
            'cp': cp,
            'trestbps':tres,
            'chol': chol,
            'fbs': fbs,
            'restecg': res,
            'thalach':tha,
            'exang':exa,
            'oldpeak':old,
            'slope':slope,
            'ca':ca,
            'thal':thal
                }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.title('Heart Disease Prediction App 🩺')
st.write("This app predicts if a patient has a heart disease.")

st.subheader('User Input Features')
st.write(input_df)

heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['target'])

combined_df = pd.concat([input_df, heart_dataset], axis=0)

df = pd.get_dummies(combined_df, columns=['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])

df = df[:1]

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

col1, col2 = st.columns(2)

with col1:
    st.subheader('Prediction')
    st.write(prediction)
    if prediction == 1:
        st.error("**Yes**, There is a higher risk of experiencing a heart disease")
    else:
        st.success("**No**, There is no risk")
with col2:
    st.image('heart.jpg', use_column_width=True)

st.subheader('Prediction Probability')
st.write(prediction_proba)
