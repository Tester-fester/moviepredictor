# app.py
import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import numpy as np

# Load model and preprocessing artifacts
model = xgb.Booster()
model.load_model('imdb_model.json')  # Make sure this file exists

certificate_encoder = joblib.load('certificate_encoder.joblib')
director_encoder = joblib.load('director_encoder.joblib')
genre_mlb = joblib.load('genre_mlb.joblib')
actor_mlb = joblib.load('actor_mlb.joblib')
preprocessing_data = joblib.load('preprocessing_data.joblib')

st.title("IMDB Rating Predictor")

with st.form("movie_form"):
    # Input fields
    runtime = st.text_input("Runtime (e.g., '120 min')", help="Enter duration like '120 min'")
    released_year = st.text_input("Released Year", help="e.g., 2023")
    certificate = st.selectbox("Certificate", certificate_encoder.categories_[0])
    director = st.text_input("Director", help="Full name as in dataset")
    genre = st.text_input("Genre (comma-separated)", help="e.g., Action, Drama")
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
            'Actors': [s.strip() for s in stars if s.strip()]
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
        final_df = pd.concat([
            df_input[['Runtime', 'Released_Year', 'Director_Avg_Rating']].astype(np.float32),
            cert_encoded,
            director_encoded,
            genre_encoded,
            actors_encoded
        ], axis=1).reindex(columns=preprocessing_data['feature_names'], fill_value=0)

        # Convert to DMatrix with feature names
        dmatrix = xgb.DMatrix(
            data=final_df.values,
            feature_names=final_df.columns.tolist(),
            nthread=-1
        )

        # Predict
        prediction = model.predict(dmatrix)[0]
        st.success(f"**Predicted IMDB Rating:** {prediction:.1f}/10")
        
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
