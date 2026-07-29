import os
import sys
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, jsonify
from src.predict_deforestation import predict_deforestation_risk

app = Flask(__name__)
# app.py - Add these routes

@app.route('/dashboard')
def dashboard():
    """Serve the dashboard HTML page"""
    return render_template('dashboard.html')

@app.route('/api/dashboard-data')
def get_dashboard_data():
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, '..', 'data', 'engineered', 'deforestation_features.csv')
        df = pd.read_csv(csv_path)
        
        # --- FIX: Handle invalid dates ---
        # Try to convert dates, coerce errors to NaT (Not a Time)
        df['Observation_Date'] = pd.to_datetime(df['Observation_Date'], errors='coerce')
        
        # Drop rows with invalid dates (like "31-13-2024")
        df = df.dropna(subset=['Observation_Date'])
        # ----------------------------------
        
        # Region columns
        region_cols = ['Region_Central', 'Region_Eastern', 'Region_Northern', 'Region_Western']
        region_names = ['Central', 'Eastern', 'Northern', 'Western']
        
        # 1. Risk Distribution
        risk_counts = df['Deforestation_Risk'].value_counts().reset_index()
        risk_counts.columns = ['Risk', 'Count']
        
        # 2. Loss by Region
        region_data = []
        for col, name in zip(region_cols, region_names):
            region_df = df[df[col] == 1]
            if len(region_df) > 0:
                region_data.append({'Region': name, 'Avg_Loss': region_df['Tree_Cover_Loss_ha'].mean()})
        region_loss = pd.DataFrame(region_data)
        
        # 3. Loss by Forest Type
        forest_cols = ['Forest_Type_Dry Forest', 'Forest_Type_Mangrove', 'Forest_Type_Montane Forest', 
                       'Forest_Type_Plantation', 'Forest_Type_Tropical Rainforest', 'Forest_Type_Woodland']
        forest_names = ['Dry Forest', 'Mangrove', 'Montane Forest', 'Plantation', 'Tropical Rainforest', 'Woodland']
        
        forest_data = []
        for col, name in zip(forest_cols, forest_names):
            forest_df = df[df[col] == 1]
            if len(forest_df) > 0:
                forest_data.append({'Forest_Type': name, 'Avg_Loss': forest_df['Tree_Cover_Loss_ha'].mean()})
        forest_loss = pd.DataFrame(forest_data)
        
        # 4. Loss Rate by Risk
        df['Loss_Rate'] = (df['Tree_Cover_Loss_ha'] / df['Forest_Area_ha'] * 100)
        risk_loss_rate = df.groupby('Deforestation_Risk')['Loss_Rate'].mean().reset_index()
        risk_loss_rate.columns = ['Risk', 'Loss_Rate']
        
        # 5. Illegal Logging
        logging_loss = df[df['Illegal_Logging'].isin(['Yes', 'No'])].groupby('Illegal_Logging')['Tree_Cover_Loss_ha'].mean().reset_index()
        logging_loss.columns = ['Illegal_Logging', 'Tree_Cover_Loss_ha']
        
        # 6. Time Trend - Extract year from cleaned dates
        df['Year'] = df['Observation_Date'].dt.year
        yearly_loss = df.groupby('Year')['Tree_Cover_Loss_ha'].mean().reset_index()
        
        # 7. Fire by Region
        fire_data = []
        for col, name in zip(region_cols, region_names):
            region_df = df[df[col] == 1]
            if len(region_df) > 0:
                fire_data.append({'Region': name, 'Fire_Incidents': region_df['Fire_Incidents'].mean()})
        fire_region = pd.DataFrame(fire_data)
        
        # 8. Correlation Matrix
        corr_cols = ['Forest_Area_ha', 'Tree_Cover_Loss_ha', 'Annual_Rainfall_mm', 
                     'Population_Density', 'Fire_Incidents']
        corr_matrix = df[corr_cols].corr().round(2)
        
        return jsonify({
            'risk_distribution': risk_counts.to_dict('records'),
            'region_loss': region_loss.to_dict('records'),
            'forest_loss': forest_loss.to_dict('records'),
            'risk_loss_rate': risk_loss_rate.to_dict('records'),
            'logging_loss': logging_loss.to_dict('records'),
            'yearly_loss': yearly_loss.to_dict('records'),
            'fire_region': fire_region.to_dict('records'),
            'correlation_matrix': corr_matrix.to_dict()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def index():
    # Only serve the HTML page here
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    # Receive JSON data from the frontend fetch request
    data = request.json
    
    observation_date = data.get('Observation_Date')
    region = data.get('Region')
    district = data.get('District')
    forest_type = data.get('Forest_Type')
    forest_area = float(data.get('Forest_Area_ha'))
    tree_cover_loss = float(data.get('Tree_Cover_Loss_ha'))
    annual_rainfall = float(data.get('Annual_Rainfall_mm'))
    population_density = int(data.get('Population_Density'))
    fire_incidents = int(data.get('Fire_Incidents'))
    illegal_logging = data.get('Illegal_Logging')

    # Run the model
    prediction = predict_deforestation_risk(
        observation_date,
        region,
        district,
        forest_type,
        forest_area,
        tree_cover_loss,
        annual_rainfall,
        population_density,
        fire_incidents,
        illegal_logging
    )

    # Return the result as JSON so the page doesn't reload
    return jsonify({"prediction": prediction})

@app.route('/api/test')
def test_api():
    """Simple test endpoint"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base_dir, '..', 'data', 'engineered', 'deforestation_features.csv')
        df = pd.read_csv(csv_path)
        
        # Just return basic info
        return jsonify({
            'status': 'success',
            'rows': len(df),
            'columns': df.columns.tolist(),
            'sample': df.head(2).to_dict('records')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)