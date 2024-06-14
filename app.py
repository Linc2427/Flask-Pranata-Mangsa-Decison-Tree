from joblib import dump, load
from flask import Flask, render_template, request, send_file, make_response
import pandas as pd
import numpy as np
import io

app = Flask(__name__)

# Load the trained decision tree model
model = load('decision_tree_model.joblib')  # Load your trained model here

month_mapping = {
    1: 'Kasa',
    2: 'Karo',
    3: 'Katelu',
    4: 'Kapat',
    5: 'Kalima',
    6: 'Kanem',
    7: 'Kapitu',
    8: 'Kawolu',
    9: 'Kasanga',
    10: 'Kasepuluh',
    11: 'Dhesta',
    12: 'Sadha'
}

global_df_predictions = pd.DataFrame()

def preprocess_data(file):
    df = pd.read_csv(file)
    
    # Menerapkan mapping untuk kolom 'Musim' menjadi 'Cuaca'
    mapping = {'Hujan': 1, 'Kemarau': 0}
    def convert_to_integer(word):
        return mapping.get(word, -1)
    df['Cuaca'] = df['Musim'].apply(convert_to_integer)
    
    # Normalisasi kolom-kolom lain
    kolom_normalisasi = ['RR_sum', 'Tavg_dasarian', 'RH_avg_dasarian', 'ff_avg_dasarian', 'wind_degrees_new']
    for kolom in kolom_normalisasi:
        min_val = min(df[kolom])
        max_val = max(df[kolom])
        df[f'Norm_{kolom}'] = [(x - min_val) / (max_val - min_val) for x in df[kolom]]
    
    # Pastikan semua kolom yang dibutuhkan ada di DataFrame yang dihasilkan
    required_columns = ['Norm_RR_sum', 'Norm_Tavg_dasarian', 'Norm_RH_avg_dasarian', 'Norm_ff_avg_dasarian', 'Norm_wind_degrees_new', 'Cuaca', 'Mangsa', 'Month_Year']
    df = df[required_columns]
    
    return df

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global global_df_predictions
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = preprocess_data(file)

            # Separate features and target
            X = df[['Norm_RR_sum', 'Norm_Tavg_dasarian', 'Norm_RH_avg_dasarian', 'Norm_ff_avg_dasarian', 'Norm_wind_degrees_new', 'Cuaca', 'Mangsa']]
            y_pred = model.predict(X)

            # Add predictions to DataFrame
            df['Bulan'] = y_pred
            df['Mangsa'] = df['Bulan'].map(month_mapping)
            df_predictions = df[['Month_Year', 'Mangsa']]
            global_df_predictions = df_predictions

            # Render results template with classification results
            return render_template('index.html', tables=[df_predictions.to_html(classes='data', index=False)], titles=df_predictions.columns.values)
    return render_template('index.html')

@app.route('/debug_save')
def debug_save_file():
    global global_df_predictions

    # Save DataFrame to CSV for debugging
    global_df_predictions.to_csv('debug_predictions.csv', index=False)

    return 'Debug CSV saved'


@app.route('/download')

def download_file():
    global global_df_predictions


    # Convert DataFrame to CSV
    output = io.StringIO()
    global_df_predictions.to_csv(output, index=False)
    output.seek(0)

    # Create a response with CSV data
    response = make_response(output.getvalue())
    response.headers['Content-Disposition'] = 'attachment; filename=prediction_results.csv'
    response.mimetype = 'text/csv'

    return response

if __name__ == '__main__':
    app.run(debug=True)
