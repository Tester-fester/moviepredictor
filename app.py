import streamlit as st
import pandas as pd
import joblib
import numpy as np

from xgboost import Booster

# Load model using Booster (avoids sklearn compatibility issues)
model = Booster()
model.load_model('imdb_model.json')

certificate_encoder = joblib.load('certificate_encoder.joblib')
director_encoder = joblib.load('director_encoder.joblib')
genre_mlb = joblib.load('genre_mlb.joblib')
actor_mlb = joblib.load('actor_mlb.joblib')
preprocessing_data = joblib.load('preprocessing_data.joblib')

st.title("IMDB Rating Predictor")

with st.form("movie_form"):
    # Input fields
    runtime = st.text_input("Runtime (e.g., '120 min')")
    released_year = st.text_input("Released Year")
    certificate = st.selectbox("Certificate", certificate_encoder.categories_[0])
    director = st.text_input("Director")
    genre = st.text_input("Genre (comma-separated, e.g., 'Action, Drama')")
    stars = [st.text_input(f"Star {i+1}") for i in range(4)]
    
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Preprocess inputs
        df_input = pd.DataFrame([{
            'Runtime': int(runtime.replace(' min', '')),
            'Released_Year': int(float(released_year)) if released_year else preprocessing_data['median_year'],
            'Certificate': certificate,
            'Director': director,
            'Genre': genre.split(', ') if genre else [],
            'Actors': [s for s in stars if s]
        }])

        # Feature engineering
        df_input['Director_Avg_Rating'] = df_input['Director'].map(
            lambda x: preprocessing_data['director_avg_rating'].get(
                x, 
                preprocessing_data['global_avg_rating']
            )
        )

        # Encode categorical features
        cert_encoded = pd.DataFrame(
            certificate_encoder.transform(df_input[['Certificate']]),
            columns=certificate_encoder.get_feature_names_out(['Certificate'])
        )
        
        director_encoded = pd.DataFrame(
            director_encoder.transform(df_input[['Director']]),
            columns=director_encoder.get_feature_names_out(['Director'])
        )

        genre_encoded = pd.DataFrame(
            genre_mlb.transform(df_input['Genre']),
            columns=genre_mlb.classes_
        )

        actors_encoded = pd.DataFrame(
            actor_mlb.transform(df_input['Actors']),
            columns=actor_mlb.classes_
        )

        # Combine features
        # Combine features
final_df = pd.concat([
    df_input[['Runtime', 'Released_Year', 'Director_Avg_Rating']],
    cert_encoded,
    director_encoded,
    genre_encoded,
    actors_encoded
], axis=1).reindex(columns=preprocessing_data['feature_names'], fill_value=0)

# Convert to DMatrix with explicit feature names
dtest = xgb.DMatrix(
    final_df.values.astype(np.float32),  # Ensure numeric type
    feature_names=final_df.columns.tolist()
)

# Predict
prediction = model.predict(dtest)[0]
st.success(f"Predicted IMDB Rating: {prediction:.2f}")
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
