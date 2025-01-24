from flask import Flask, request, render_template
import pandas as pd
import joblib
from xgboost import XGBRegressor
import numpy as np

app = Flask(__name__)

# Load model and preprocessing artifacts
model = XGBRegressor()
model.load_model('imdb_model.json')

certificate_encoder = joblib.load('certificate_encoder.joblib')
director_encoder = joblib.load('director_encoder.joblib')
genre_mlb = joblib.load('genre_mlb.joblib')
actor_mlb = joblib.load('actor_mlb.joblib')
preprocessing_data = joblib.load('preprocessing_data.joblib')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get inputs from form
            runtime = request.form['runtime']
            released_year = request.form['released_year']
            certificate = request.form['certificate']
            director = request.form['director']
            genre = request.form['genre']
            stars = [request.form[f'star{i+1}'] for i in range(4)]

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
            final_df = pd.concat([
                df_input[['Runtime', 'Released_Year', 'Director_Avg_Rating']],
                cert_encoded,
                director_encoded,
                genre_encoded,
                actors_encoded
            ], axis=1).reindex(columns=preprocessing_data['feature_names'], fill_value=0)

            # Predict
            prediction = model.predict(final_df)
            return f"Predicted IMDB Rating: {prediction[0]:.2f}"
        
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Render HTML form for GET requests
    return '''
    <form method="post">
        Runtime (e.g., 120 min): <input type="text" name="runtime"><br>
        Released Year: <input type="text" name="released_year"><br>
        Certificate: <input type="text" name="certificate"><br>
        Director: <input type="text" name="director"><br>
        Genre (comma-separated): <input type="text" name="genre"><br>
        Star 1: <input type="text" name="star1"><br>
        Star 2: <input type="text" name="star2"><br>
        Star 3: <input type="text" name="star3"><br>
        Star 4: <input type="text" name="star4"><br>
        <input type="submit" value="Predict">
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)
