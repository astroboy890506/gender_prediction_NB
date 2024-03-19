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

def extract_gender_features(name):           # step 3: feature extraction (since the data is clean, we may skip the step 2: data cleaning)
    name = name.lower()
    features = {}
    features["suffix"] = name[-1:]
    features["suffix2"] = name[-2:] if len(name) > 1 else name[0]
    features["suffix3"] = name[-3:] if len(name) > 2 else name[0]
    features["suffix4"] = name[-4:] if len(name) > 3 else name[0]
    features["suffix5"] = name[-5:] if len(name) > 4 else name[0]
    features["suffix6"] = name[-6:] if len(name) > 5 else name[0]
    features["prefix"] = name[:1] #J
    features["prefix2"] = name[:2] if len(name) > 1 else name[0]
    features["prefix3"] = name[:3] if len(name) > 2 else name[0]
    features["prefix4"] = name[:4] if len(name) > 3 else name[0]
    features["prefix5"] = name[:5] if len(name) > 4 else name[0]
    print (features)
    return features

def main():
    st.title("Gender Prediction App")
    name = st.write("Enter a name to predict the gender:")
    extract_gender_features(name)
    
    # Load the model
    model_path = 'gender_prediction_model.joblib'
    model = load_model(model_path)

    name_input = st.text_input("Name:")
    if name_input:
        predicted_gender = predict_gender(name_input, model)
        st.write(f"Predicted gender for {name_input}: {predicted_gender}")

if __name__ == "__main__":
    main()
