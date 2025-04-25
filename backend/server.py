# server.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('model.joblib')

# Mapping dictionaries (FIXED SYNTAX)
BOOLS_MAP = {"No": 0.0, "Yes": 1.0}
INVERTER_MANUFACTURER_MAP = {"SolarEdge": 1.0, "Enphase": 2.0, "SMA": 3.0, "GoodWe": 4.0}
ARRAY_TYPE_MAP = {"Roof Mount": 0.0, "Ground Mount": 1.0}
TRUSS_RAFTER_MAP = {"Truss": 0.0, "Rafter": 1.0}
INTERCONNECTION_TYPE_MAP = {
    "A1": 0, "A2": 1, "A3": 2, "A4": 3, "A*": 4,
    "B1": 5, "B2": 6, "B*": 7, "C1": 8, "C2": 9, "C3": 10, "C*": 11
}  # Added closing brace
ROOF_TYPE_MAP = {
    "Asphalt Shingles": 0, "Standing Seam Metal Roof": 1, "Ag Metal": 2,
    "EPDM (Flat Roof)": 3, "Ground Mount": 4
}  # Added closing brace
ATTACHMENT_TYPE_MAP = {
    "Flashfoot 2": 0, "Unknown": 1, "S-5!": 2, "Ejot": 3, "Ground Mount": 4,
    "RT Mini": 5, "Flashview": 6, "Hugs": 7
}  # Added closing brace
PORTRAIT_LANDSCAPE_MAP = {"Portrait": 0, "Landscape": 1, "Both": 2}
INSTALL_SEASON_MAP = {"Spring": 0, "Summer": 1, "Fall": 2, "Winter": 3}

@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting time..")
    try:
        input_data = request.json
        print(input_data)
        # Map categorical fields
        input_data['Inverter Manufacturer'] = INVERTER_MANUFACTURER_MAP.get(input_data['Inverter Manufacturer'], 0.0)
        input_data['Array Type'] = ARRAY_TYPE_MAP.get(input_data['Array Type'], 0.0)
        input_data['Truss / Rafter'] = TRUSS_RAFTER_MAP.get(input_data['Truss / Rafter'], 0.0)
        input_data['Squirrel Screen'] = BOOLS_MAP.get(input_data['Squirrel Screen'], 0.0)
        input_data['Consumption Monitoring'] = BOOLS_MAP.get(input_data['Consumption Monitoring'], 0.0)
        input_data['Reinforcements'] = BOOLS_MAP.get(input_data['Reinforcements'], 0.0)
        input_data['Rough Electrical Inspection'] = BOOLS_MAP.get(input_data['Rough Electrical Inspection'], 0.0)
        input_data['Interconnection Type'] = INTERCONNECTION_TYPE_MAP.get(input_data['Interconnection Type'], 0)
        input_data['Roof Type'] = ROOF_TYPE_MAP.get(input_data['Roof Type'], 0)
        input_data['Attachment Type'] = ATTACHMENT_TYPE_MAP.get(input_data['Attachment Type'], 1)
        input_data['Portrait / Landscape'] = PORTRAIT_LANDSCAPE_MAP.get(input_data['Portrait / Landscape'], 0)
        input_data['Install Season'] = INSTALL_SEASON_MAP.get(input_data['Install Season'], 0)

        # Add missing columns with default values
        input_data['Estimated # of Salaried Employees on Site'] = 0
        input_data['Estimated Total # of People on Site'] = 0
        input_data['Total # of Days on Site'] = 0
        input_data['Total # Hourly Empoyees on Site'] = 0

        # Create DataFrame for prediction
        df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(df)
        print("Done! Prediction =", prediction[0])
        return jsonify({'prediction': float(prediction[0]/60)})

    except Exception as e:
        print("Prediction error:", str(e))
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    print("Starting backend...")
    app.run(debug=True)
