import streamlit as st
import joblib

# Load the trained model from the .joblib file
def load_model(model_path):
    model = joblib.load(model_path)
    return model

def predict_gender(name_input, model):
    features = extract_gender_features(name_input)  # Use the same feature extraction
    gender = model.classify(features)
    return gender

def main():
    st.title("Gender Prediction App")
    st.write("Enter a name to predict the gender:")

    # Load the model
    model_path = 'gender_prediction_model.joblib'
    model = load_model(model_path)

    name_input = st.text_input("Name:")
    if name_input:
        predicted_gender = predict_gender(name_input, model)
        st.write(f"Predicted gender for {name_input}: {predicted_gender}")

if __name__ == "__main__":
    main()