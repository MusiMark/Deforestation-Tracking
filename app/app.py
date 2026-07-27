import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, render_template, jsonify
from src.predict_deforestation import predict_deforestation_risk

app = Flask(__name__)

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

if __name__ == '__main__':
    app.run(debug=True)