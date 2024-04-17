# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import json

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
            '%d/%m/%Y %H:%M:%S'), "value": pred}
        for date, pred in zip(label_map, predictions_list)
    ]

    # Convert the list to a JSON string
    predictions_data_json = json.dumps(date_prediction_pairs, indent=4)

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
    date_prediction_pairs = [{"date": date, "value": pred}
                             for date, pred in zip(future_dates, predictions)]

    # Convert list of dictionaries to JSON string
    predict_intervals_json = json.dumps(date_prediction_pairs, indent=4)

    return predict_intervals_json

# This function returns a JSON object consisting of two merged JSON objects.


def merge_json_outputs(json1, json2):
    # Parse the JSON strings into Python lists
    data1 = json.loads(json1)
    data2 = json.loads(json2)

    # Merge the two lists
    merged_data = data1 + data2

    # Convert the list to a JSON string
    final_data_json = json.dumps(merged_data, indent=4)

    # Convert the merged list back to a JSON string
    return final_data_json

# Plotting Functions


def plot_data(merged_json):
    # Parse the JSON string back into a Python list
    data = json.loads(merged_json)

    # Convert the list to a DataFrame
    df = pd.DataFrame(data)

    # Ensure 'date' column is a datetime type
    df['date'] = pd.to_datetime(df['date'])

    # Sort the DataFrame by date
    df.sort_values('date', inplace=True)

    # Plotting
    plt.figure(figsize=(10, 5))
    plt.plot(df['date'], df['value'], marker='o', linestyle='-')
    plt.title('Predict days')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_results_from_json(real_data_json, predictions_json):
    # Parse the JSON strings into Python dictionaries
    real_data = json.loads(real_data_json)
    predictions = json.loads(predictions_json)

    # Convert dictionaries to pandas DataFrames
    real_df = pd.DataFrame(real_data)
    predictions_df = pd.DataFrame(predictions)

    # Ensure 'date' column is datetime type
    real_df['date'] = pd.to_datetime(real_df['date'])
    predictions_df['date'] = pd.to_datetime(predictions_df['date'])

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(real_df['date'], real_df['value'],
             label='Actual Data', color='red', marker='o')
    plt.plot(predictions_df['date'], predictions_df['value'],
             label='Predicted Data', color='blue', marker='x')
    plt.title('Comparison of Actual and Predicted Values')
    plt.xlabel('Date')
    plt.ylabel('Values')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def main():
    # Paths to CSV file
    plant_id_source = pd.read_csv(r'data/plant_1_merged.csv')
    # Load Model
    model = load_model('Model/solar_prediction.keras')
    lag = 1
    delay = 25
    num_intervals = 15
    # Data preprocessing
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag, delay)

    # Convert real data and dates to JSON
    real_data_json = real_data_to_json(plant_id_source)
    # print(real_data_json)

    # Make predictions using the original model
    predicted_power = model.predict(X_1)

    # Generate JSON from predictions
    predictions_data_json = predictions_to_json(label_map_1, predicted_power)
    # print(predictions_data_json)

    # Get the last segment of data for prediction given n days
    last_processed_sequence = X_1[-1]
    last_date_in_dataset = label_map_1[-1]

    # Generate JSON from Predictions given n days
    predict_days_json = predict_next_n_intervals(
        model, last_processed_sequence, last_date_in_dataset, num_intervals)
    # print(predict_days_json)

    # Join Real Data JSON with prediction by n days JSON
    final_data_json = merge_json_outputs(real_data_json, predict_days_json)
    # print(final_data_json)

    # Plotting Area
    # plot_data(predict_days_json)
    # plot_results_from_json(real_data_json, predictions_data_json)


if __name__ == "__main__":
    main()
