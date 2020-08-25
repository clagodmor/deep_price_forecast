import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from esios import *
from functools import reduce
import tensorflow as tf
import matplotlib.pyplot as plt

indicatorsDict = {'demand': 460,'price': 805,'wind':541,'solar':10034}

indicatorsItems = indicatorsDict.items()

start_date = datetime.date.today() - datetime.timedelta(days=5)
end_date = datetime.date.today()
start_ = start_date.strftime("%Y-%m-%d") + 'T00:00:00'
end_ = end_date.strftime("%Y-%m-%d") + 'T23:50:00'
token = '6cc21e0b60e9931e7522a6ce72a1a09f3a6fadc6f08b142f956db142c6858bc2'    # Introduce ESIOS token
esios = ESIOS(token)
country = 'Spain' #Spain, France or Portugal are the options

for indicatorName, indicatorValue in indicatorsItems:
    print ('Start Date: ' + start_date.strftime("%Y-%m-%d"))
    print ('End Date: ' + end_date.strftime("%Y-%m-%d"))
    indicators_ = list()
    indicators_.append(indicatorValue)
    dfmul , df_list, names = esios.get_multiple_series(indicators_, start_, end_, country)
    df = dfmul[names]
    df = df.reset_index()
    df.columns = ['Date', indicatorName]
    df['Date'] = df['Date'].str.replace('.', ' ')
    df['Date'] = df['Date'].str.split().str[0]
    df['Date'] = df['Date'].str.replace('T', ' ')
    # Export to .csv file
    try:
        os.stat("Files/")
    except:
        os.mkdir("Files/")
    df.to_csv(path_or_buf= 'Files/' + str(indicatorName) + '.csv', sep='^', index=False)
    print('Generated:' + str(indicatorName))

    with open('Files/demand.csv', 'r') as fichero:
        dfDemand = pd.read_csv(fichero, sep='^', dtype='object')
        dfDemand['Date'] = pd.to_datetime(dfDemand['Date'])

    with open('Files/solar.csv', 'r') as fichero:
        dfSolar = pd.read_csv(fichero, sep='^', dtype='object')
        dfSolar['Date'] = pd.to_datetime(dfSolar['Date'])

    with open('Files/wind.csv', 'r') as fichero:
        dfWind = pd.read_csv(fichero, sep='^', dtype='object')
        dfWind['Date'] = pd.to_datetime(dfWind['Date'])

    with open('Files/price.csv', 'r') as fichero:
        dfPrice = pd.read_csv(fichero, sep='^', dtype='object')
        dfPrice['Date'] = pd.to_datetime(dfPrice['Date'])

dfDemand = dfDemand.drop_duplicates(subset='Date', keep='first')
dfSolar = dfSolar.drop_duplicates(subset='Date', keep='first')
dfWind = dfWind.drop_duplicates(subset='Date', keep='first')
dfPrice = dfPrice.drop_duplicates(subset='Date', keep='first')

df = [dfDemand, dfSolar, dfWind, dfPrice]
df = reduce(lambda left,right: pd.merge(left,right,on='Date', how = 'right'), df)

df['demand'] = df['demand'].astype(float)
df['solar'] = df['solar'].astype(float)
df['wind'] = df['wind'].astype(float)
df['price'] = df['price'].astype(float)


# MODELO
features_considered = ['demand', 'solar', 'wind', 'price']
features = df[features_considered]
features.index = df['Date']

TRAIN_SPLIT = 40000
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

# Create a new model instance
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32,
                               return_sequences=True,
                               input_shape=(120, 4)))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(12))

# Load the previously saved weights
checkpoint_path = "models/multi_step_model.ckpt"
model.load_weights(checkpoint_path)

# Re-evaluate the model
def create_time_steps(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

def multi_step_plot(history, true_future, prediction):
  plt.figure(figsize=(12, 6))
  num_in = create_time_steps(len(history))
  num_out = len(true_future)

  plt.plot(num_in, np.array(history[:, 3]), label='History')
  plt.plot(np.arange(num_out)/1, np.array(true_future), 'bo',
           label='True Future')
  if prediction.any():
    plt.plot(np.arange(num_out)/1, np.array(prediction), 'ro',
             label='Predicted Future')
  plt.legend(loc='upper left')
  plt.show()

for x, y in dataset.take(3):
  multi_step_plot(x[0], y[0], dataset.predict(x)[0])
mae = model.evaluate(dataset, steps=1000)
print("MAE for LSTM is %.4f" %mae)