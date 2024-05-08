from flask import Flask, jsonify, request, render_template
import pandas as pd
from datetime import datetime
import joblib
from pipe import FullPipeline1 , LabelEncodeColumns ,CustomOneHotEncoder , DropColumnsTransformer ,OutlierThresholdTransformer ,DateExtractor, DataFrameImputer,StandardScaleTransform

app = Flask(__name__)

# Load the trained model
model = joblib.load('one_hot_model.pkl')

# Load the preprocessing pipeline
pipeline = joblib.load('one_hot_pipeline.pkl')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['GET'])
def predict_api():
    # Get query parameters from the request
    date_time = request.args.get('date_time')
    wind_speed = float(request.args.get('wind_speed'))
    theoretical_power = float(request.args.get('theoretical_power'))
    wind_direction = float(request.args.get('wind_direction'))

    original_datetime = datetime.strptime(date_time, "%Y-%m-%dT%H:%M")
    formatted_datetime_str = original_datetime.strftime("%d %m %Y %H:%M")

    # Create a DataFrame with the query parameters
    data = pd.DataFrame({'Date/Time': [formatted_datetime_str],
                         'Wind Speed (m/s)': [wind_speed],
                         'Theoretical_Power_Curve (KWh)': [theoretical_power],
                         'Wind Direction (°)': [wind_direction]
                         })

    # Preprocess data using the pipeline
    transformed_data = pipeline.transform(data)
    
    # Predict LV Active Power using the model
    lv_active_power = model.predict(transformed_data)[0]

    return jsonify({'lv_active_power': lv_active_power})
def predict_404(date_time, wind_speed, theoretical_power, wind_direction):
    original_datetime = datetime.strptime(date_time, "%Y-%m-%dT%H:%M")
    formatted_datetime_str = original_datetime.strftime("%d %m %Y %H:%M")

    # Create a DataFrame with the query parameters
    data = pd.DataFrame({'Date/Time': [formatted_datetime_str],
                         'Wind Speed (m/s)': [wind_speed],
                         'Theoretical_Power_Curve (KWh)': [theoretical_power],
                         'Wind Direction (°)': [wind_direction]
                         })

    # Preprocess data using the pipeline
    transformed_data = pipeline.transform(data)
    
    # Predict LV Active Power using the model
    lv_active_power = model.predict(transformed_data)[0]

    return lv_active_power


@app.route('/result', methods=['POST'])
def predict():
    # Get data from the form
    date_time = request.form['date_time']
    wind_speed = float(request.form['wind_speed'])
    theoretical_power = float(request.form['theoretical_power'])
    wind_direction = float(request.form['wind_direction'])

    # Use predict_api() to get the predicted value
    lv_active_power = predict_404(date_time, wind_speed, theoretical_power, wind_direction)

    # Render template with the predicted value
    return render_template('result.html', lv_active_power=lv_active_power)

# @app.route('/result', methods=['POST'])
# def predict():
#     # Get data from the form
#     date_time = request.form['date_time']
#     wind_speed = float(request.form['wind_speed'])
#     theoretical_power = float(request.form['theoretical_power'])
#     wind_direction = float(request.form['wind_direction'])
    

#     original_datetime = datetime.strptime(date_time, "%Y-%m-%dT%H:%M")

#     # Convert to the desired format
#     formatted_datetime_str = original_datetime.strftime("%d %m %Y %H:%M")


#     # Create a DataFrame with the form data
#     data = pd.DataFrame({'Date/Time': [formatted_datetime_str],
#                          'Wind Speed (m/s)': [wind_speed],
#                          'Theoretical_Power_Curve (KWh)': [theoretical_power],
#                          'Wind Direction (°)': [wind_direction]
#                          })
    
#     # sample_data = {
#     # 'Date/Time': ['01 01 2018 00:10'],
#     # 'Wind Speed (m/s)': [5.672167],
#     # 'Theoretical_Power_Curve (KWh)': [519.917511],
#     # 'Wind Direction (°)': [268.641113]
#     # }
#     # df = pd.DataFrame(sample_data)
    
#     #result_html = data.to_html(index=False)
#     #f1 = FullPipeline1()
#     # Preprocess data using your pipeline
#     transformed_data = pipeline.transform(data)
#     # Predict LV Active Power using your model
#     lv_active_power = model.predict(transformed_data)
#     # print(lv_active_power)

#     return render_template('result.html', lv_active_power=lv_active_power)

if __name__ == '__main__':
    app.run(debug=True)
