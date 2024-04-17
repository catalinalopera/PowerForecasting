# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras import optimizers

plant_id_source = pd.read_csv(r'data/combined_plants_3.csv')


# Function to pre process data
# Now make this a function
def data_pre_process(plant_id, lag, delay):
    df = plant_id
    # Re arrange df
    df = df.loc[:, ['DATE_TIME', 'DC_POWER', 'AC_POWER', 'AMBIENT_TEMPERATURE',
                    'MODULE_TEMPERATURE', 'IRRADIATION', 'DAILY_YIELD']]
    # df = df.set_index('DATE_TIME')

    # Now split data into X and y
    X = df.drop(columns={'DAILY_YIELD', 'DATE_TIME'})
    y = df['DAILY_YIELD']

    # Now let's scale data
    scaler_x = MinMaxScaler()  # create scaler object
    scaled_data = scaler_x.fit_transform(X)  # fit transform data

    # Split data into samples and reshape X
    samples = [scaled_data[i:i+delay] for i in range(0, (y.shape[0]), delay)]

    # Adjust last batch of observations
    if len(samples[-1]) < delay:
        samples = np.array(samples[:-1])
    else:
        samples = np.array(samples)

    # now re shape y
    new_y = [y.iloc[i+delay+lag]
             for i in range(0, (y.shape[0]-(delay+lag)), delay)]
    # scale y
    # scaler_y = MinMaxScaler()
    new_y = np.array(new_y)
    # scaled_y = scaler_y.fit_transform(new_y.reshape(1,-1))

    # Get label map
    index = df.index
    label_map = [index[i+delay+lag]
                 for i in range(0, (y.shape[0]-(delay+lag)), delay)]

    return samples, new_y, scaler_x, lag, delay, label_map


# LSTM model
def lstm_model_init(X_train, verbose=True):
    # RNN-LSTM architechture
    keras.utils.set_random_seed(97)  # Set seed for reproducibility
    lstm_model = Sequential(name='lstm_model')
    lstm_model.add(layers.LSTM(units=70, return_sequences=True,
                               input_shape=(X_train[0].shape), activation="relu"))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(layers.LSTM(
        units=35, return_sequences=False, activation="relu"))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(layers.Dense(1))

    # Compile model
    optimizer = optimizers.Adam(learning_rate=0.001)
    lstm_model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

    if verbose == True:
        print(lstm_model.summary())

    return lstm_model


def main():
    # Assume X_1 and y_1 are already defined
    X_1, y_1, x_scaler_1, lag_1, delay_1, label_map_1 = data_pre_process(
        plant_id_source, lag=1, delay=25)

    lstm_model = lstm_model_init(X_1, verbose=True)

    lstm_history = lstm_model.fit(
        X_1, y_1, epochs=200, batch_size=32, validation_split=0.2)

    # Save the Model
    lstm_model.save("Model/solar_prediction.keras")

    # Plot training and validation loss
    print("\n")
    loss = lstm_history.history['loss']
    val_loss = lstm_history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.xlabel('Epochs')
    plt.title('Training and validation loss ')
    plt.legend()
    plt.show()

    # Make predictions and plot with correct values
    predicted_power = lstm_model.predict(X_1)
    plt.plot(y_1, color='red', label='Real')
    plt.plot(predicted_power, color='blue', label='Predicted')
    plt.title('Prediction')
    plt.xlabel('Time')
    plt.ylabel('Power')
    plt.legend()
    plt.show()

    # Evaluate model
    # Evaluate regression model using mse and map
    print("MSE for model: ")
    print(mean_squared_error(y_1, predicted_power))
    print("\n")
    print("MAPE for model: ")
    print(mean_absolute_percentage_error(y_1, predicted_power))


if __name__ == "__main__":
    main()
