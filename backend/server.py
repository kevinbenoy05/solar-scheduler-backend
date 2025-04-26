from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load('model.joblib')

# Categorical mappings
MAPPINGS = {
    'Inverter Manufacturer': {"SolarEdge": 1.0, "Enphase": 2.0, "SMA": 3.0, "GoodWe": 4.0},
    'Array Type': {"Roof Mount": 0.0, "Ground Mount": 1.0},
    'Truss / Rafter': {"Truss": 0.0, "Rafter": 1.0, "Grount Mount": 2.0, "Purlin": 3.0, "TJI": 4.0},
    'Squirrel Screen': {"No": 0.0, "Yes": 1.0},
    'Consumption Monitoring': {"No": 0.0, "Yes": 1.0},
    'Reinforcements': {"No": 0.0, "Yes": 1.0},
    'Rough Electrical Inspection': {"No": 0.0, "Yes": 1.0},
    'Interconnection Type': {"A1":0,"A2":1,"A3":2,"A4":3,"A*":4,"B1":5,"B2":6,"B*":7,"C1":8,"C2":9,"C3":10,"C*":11},
    'Roof Type': {"Asphalt Shingles":0,"Standing Seam Metal Roof":1,"Ag Metal":2,"EPDM (Flat Roof)":3,"Ground Mount":4},
    'Attachment Type': {"Flashfoot 2":0,"Unknown":1,"S-5!":2,"Ejot":3,"Ground Mount":4,"RT Mini":5,"Flashview":6,"Hugs":7},
    'Portrait / Landscape': {"Portrait":0,"Landscape":1,"Both":2},
    'Install Season': {"Spring":0,"Summer":1,"Fall":2,"Winter":3}
}

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.json
        print(input_data)
        # Convert numerical fields
        numerical_fields = [
            'Drive Time', 'Tilt', 'Tilt2', 'Azimuth', 'Azimuth2', 'Azimuth3',
            'Panel QTY', 'System Rating (kW DC)', 'Module Length', 'Module Width',
            'Module Weight', '# of Arrays', '# of reinforcement', '# of Stories',
            'Estimated Total # of People on Site'
        ]
        for field in numerical_fields:
            input_data[field] = float(input_data.get(field, 0)) if input_data.get(field) else 0.0

        # Map categorical fields
        for field, mapping in MAPPINGS.items():
            input_data[field] = mapping.get(input_data.get(field), 0.0)

        # Create feature dataframe
        features = [
            'Drive Time', 'Tilt', 'Tilt2', 'Azimuth', 'Azimuth2', 'Azimuth3',
            'Panel QTY', 'System Rating (kW DC)', 'Inverter Manufacturer',
            'Array Type', 'Squirrel Screen', 'Consumption Monitoring',
            'Truss / Rafter', 'Reinforcements', 'Rough Electrical Inspection',
            'Interconnection Type', 'Module Length', 'Module Width',
            'Module Weight', '# of Arrays', '# of reinforcement', 'Roof Type',
            'Attachment Type', 'Portrait / Landscape', '# of Stories',
            'Install Season', 'Estimated Total # of People on Site'
        ]
        df = pd.DataFrame({k: [input_data.get(k, 0)] for k in features})

        # Make prediction
        prediction = model.predict(df)[0]
        return jsonify({'prediction': float(prediction/60)})  # Convert back to hours

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
