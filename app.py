from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json
from datetime import datetime
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)

# Global Parameters for the application
plant_id_source = None  # Store the input file
num_days = 0  # Store num of days entered by the user
model = load_model('Model/solar_prediction.keras')
lag = 1
delay = 25

# Model functions


def data_pre_process(plant_id_source, lag, delay):
    # Data Selection and Indexing
    df = plant_id_source
    df = df.loc[:, ['DATE_TIME', 'DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE',
                    'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']]
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)
    # Feature Selection and Scaling
    X = df.drop(columns=['DAILY_YIELD'])
    y = df['DAILY_YIELD']
    scaler_x = MinMaxScaler()
    scaled_data = scaler_x.fit_transform(X)
    # Sample Preparation
    samples = [scaled_data[i:i + delay] for i in range(0, y.shape[0] - delay)]
    if len(samples[-1]) < delay:
        samples.pop()  # Remove the last sample if it's not of full length
    # Target Preparation
    new_y = np.array([y.iloc[i + delay + lag - 1]
                     for i in range(y.shape[0] - delay - lag)])
    # Label Mapping
    label_map = [df.index[i + delay + lag - 1]
                 for i in range(y.shape[0] - delay - lag)]

    return np.array(samples), new_y, scaler_x, lag, delay, label_map

# This function returns a JSON object containing predictions over time.


def predictions_to_json(label_map, predictions_data):
    # Convert predictions to a list
    predictions_list = predictions_data.flatten().tolist()

    # Prepare a dictionary with dates and predictions
    date_prediction_pairs = [
        {"date": pd.to_datetime(date).strftime(
            '%d/%m/%Y %H:%M:%S'), "value": value}
        for date, value in zip(label_map, predictions_list)
    ]

    # Convert the list to a JSON string
    predictions_data_json = json.dumps(date_prediction_pairs)

    return predictions_data_json

# This function returns a JSON object containing real data over time.


def real_data_to_json(plant_id_source):
    data = plant_id_source

    # Convert 'DATE_TIME' to datetime
    data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])

    # Convert real data to a list
    real_data_list = data['DAILY_YIELD'].tolist()
    label_map = data['DATE_TIME']

    # Prepare a dictionary with dates and real data
    date_real_data_pairs = [
        {"date": date.strftime('%d/%m/%Y %H:%M:%S'), "value": value}
        for date, value in zip(label_map, real_data_list)
    ]

    # Convert the list to a JSON string
    real_data_json = json.dumps(date_real_data_pairs, indent=4)

    return real_data_json


# This function returns a JSON object containing predictions for the number of intervals specified by the user.

def predict_next_n_intervals(lstm_model, last_segment_scaled, last_date_in_dataset, num_intervals):
    predictions = []
    current_batch = np.expand_dims(last_segment_scaled, axis=0)

    # Extract only the date part, normalize to midnight, and add one day
    last_date_only = pd.to_datetime(
        last_date_in_dataset).normalize() + pd.Timedelta(days=1)

    for _ in range(num_intervals):
        # Generates the next output
        next_prediction = lstm_model.predict(current_batch)
        predictions.append(float(next_prediction.flatten()[0]))

        # Update the batch for the next prediction
        new_features = np.copy(current_batch[:, -1, :])
        new_features[0, -1] = next_prediction
        current_batch = np.append(current_batch[:, 1:, :], [
                                  new_features], axis=1)

    # Generating future dates every 15 minutes starting from midnight of the next day from the last date
    future_dates = pd.date_range(
        start=last_date_only, periods=num_intervals, freq='15T')
    future_dates = [date.strftime('%d/%m/%Y %H:%M:%S')
                    for date in future_dates]

    # Combine dates and predictions into a list of dictionaries
    date_prediction_pairs = [{"date": date, "value": value}
                             for date, value in zip(future_dates, predictions)]

    # Convert list of dictionaries to JSON string
    predict_intervals_json = json.dumps(date_prediction_pairs)

    return predict_intervals_json


# This function returns a JSON object consisting of two merged JSON objects.


def merge_json_outputs(json1, json2):
    # Parse the JSON strings into Python lists
    data1 = json.loads(json1)
    data2 = json.loads(json2)

    # Merge the two lists
    merged_data = data1 + data2

    # Convert the list to a JSON string
    final_data_json = json.dumps(merged_data)

    # Convert the merged list back to a JSON string
    return final_data_json


# Flask Functions


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/Dashboard')
def Dashboard():
    return render_template('dashboard.html')


@app.route('/input_data')
def input_data():
    return render_template('upload_data.html')


@app.route('/form_num_intervals')
def form_num_intervals():
    return render_template('input_prediction.html')


@app.route('/plotreal')
def plotreal():
    real_data_json = real_data_to_json(plant_id_source)
    # Convert JSON string back to a Python list of dictionaries to display in html
    plotdata = json.loads(real_data_json)
    return render_template('plot_real_data.html', plotdata=plotdata)


@app.route('/plotpredictionintervals')
def plotpredictionintervals():
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)
    last_processed_sequence = X_1[-1]
    last_date_in_dataset = label_map_1[-1]
    predict_days_json = predict_next_n_intervals(
        model, last_processed_sequence, last_date_in_dataset, num_intervals)
    # Convert JSON string back to a Python list of dictionaries to display in html
    plotdata = json.loads(predict_days_json)
    return render_template('plot_prediction_data.html', plotdata=plotdata)


@app.route('/plotdatatogether', methods=['GET', 'POST'])
def plotdatatogether():
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)
    real_data_json = real_data_to_json(plant_id_source)
    last_processed_sequence = X_1[-1]
    last_date_in_dataset = label_map_1[-1]
    predict_days_json = predict_next_n_intervals(
        model, last_processed_sequence, last_date_in_dataset, num_intervals)
    data1 = json.loads(real_data_json)
    data2 = json.loads(predict_days_json)

    # Calculate the last 7 days range
    end_date = datetime.strptime(data2[-1]['date'], '%d/%m/%Y %H:%M:%S')
    start_date = end_date - timedelta(days=7)

    # Filter data for the last 7 days
    filtered_data1 = [item for item in data1 if datetime.strptime(
        item['date'], '%d/%m/%Y %H:%M:%S') >= start_date]
    filtered_data2 = [item for item in data2 if datetime.strptime(
        item['date'], '%d/%m/%Y %H:%M:%S') >= start_date]

    return render_template('plot_data_together.html', data1=filtered_data1, data2=filtered_data2)


@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global plant_id_source
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and file.filename.endswith('.csv'):
        # Read the file into a DataFrame
        plant_id_source = pd.read_csv(file)
        # Redirect back to the upload form or to another page
        return redirect(url_for('Dashboard'))
    return "Invalid file type", 400


@app.route('/predictions')
def show_predictions():
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)
    predicted_power = model.predict(X_1)
    predictions_data_json = predictions_to_json(label_map_1, predicted_power)
    # Convert JSON string back to a Python list of dictionaries
    predictions_data_json = json.loads(predictions_data_json)
    # Pagination logic for loading - better performace in frontend
    page = request.args.get('page', 1, type=int)
    # Change per_page parameter to indicate number of rows per page
    per_page = 40
    total_pages = len(predictions_data_json) // per_page + \
        (1 if len(predictions_data_json) % per_page > 0 else 0)
    displayed_data = predictions_data_json[(page-1)*per_page: page*per_page]
    return render_template('display_predict_data.html', json_data=displayed_data, title="Display Predictions", page=page, total_pages=total_pages)


@app.route('/predict_next_intervals', methods=['GET'])
def predict_next_intervals():
    global num_intervals
    # Take the value entered by the user through the HTML form and send it using the GET method.
    num_intervals = request.args.get('num_intervals', type=int)
    if num_intervals is None:
        # Bad request if no num_days provided
        return "Please specify the number of days for prediction.", 400
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)
    last_processed_sequence = X_1[-1]
    last_date_in_dataset = label_map_1[-1]
    predict_days_json = predict_next_n_intervals(
        model, last_processed_sequence, last_date_in_dataset, num_intervals)
    # Convert JSON string back to a Python list of dictionaries to display in html
    predict_days_json = json.loads(predict_days_json)
    return render_template('display_data.html', json_data=predict_days_json, title="Predict the Next Intervals")


@app.route('/real_data')
def show_real_data():
    real_data_json = real_data_to_json(plant_id_source)
    # Convert JSON string back to a Python list of dictionaries to display in html
    real_data_json = json.loads(real_data_json)
    # Pagination logic for loading - better performace in frontend
    page = request.args.get('page', 1, type=int)
    # Change per_page parameter to indicate number of rows per page
    per_page = 40
    total_pages = len(real_data_json) // per_page + \
        (1 if len(real_data_json) % per_page > 0 else 0)
    displayed_data = real_data_json[(page-1)*per_page: page*per_page]

    return render_template('display_real_data.html', json_data=displayed_data, title="Display Real Data", page=page, total_pages=total_pages)


@app.route('/merge_json')
def merge_jsons():
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)
    real_data_json = real_data_to_json(plant_id_source)
    last_processed_sequence = X_1[-1]
    last_date_in_dataset = label_map_1[-1]
    predict_days_json = predict_next_n_intervals(
        model, last_processed_sequence, last_date_in_dataset, num_intervals)
    final_data_json = merge_json_outputs(real_data_json, predict_days_json)
    # Convert JSON string back to a Python list of dictionaries to display in html
    final_data_json = json.loads(final_data_json)
    return render_template('display_data.html', json_data=final_data_json, title="Display Real and Predictions Data")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
