

try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

import os
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import streamlit as st

csv_path= 'DATA/Data_new.csv'
df = pd.read_csv(csv_path, sep='^')
df.head()

date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

columns = ['demand', 'solar', 'wind', 'price','Day sin', 'Day cos', 'Year sin','Year cos']
features = df[columns]
TRAIN_SPLIT=40000
dataset = features.values
data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
data_std = dataset[:TRAIN_SPLIT].std(axis=0)
dataset = (dataset-data_mean)/data_std

def multivariate_window(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

past_history = 120
future_target = 12
STEP = 1
SINGLE_STEP = False
BATCH_SIZE = 256


x_val_multi, y_val_multi = multivariate_window(dataset=dataset, target=dataset[:, 3],
  start_index=TRAIN_SPLIT, end_index=None, history_size=past_history,
  target_size=future_target, step=STEP, single_step=SINGLE_STEP)

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

def time_steps_creation(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(32,return_sequences=True,
                               input_shape=(120, 4)))
model.add(tf.keras.layers.LSTM(16, activation='relu'))
model.add(tf.keras.layers.Dense(12))

checkpoint_path = "models/multi_step_model.ckpt"
model.load_weights(checkpoint_path)



st.write("""
# Spain Electrical Price forecast

The table below shows the prediction of the electrical price forecast. 
It is calculating using a window of 5 days, and predicts 12 hours from the moment you run it. """)

progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)

for x, y in val_data_multi.take(3):
  num_in = time_steps_creation(len(x[0]))
  num_out = len(y[0])
  chart = st.line_chart(num_in, np.array(x[0][:, 3]))
  chart.add_rows(np.arange(num_out) / STEP, np.array(y[0]))

  if model.predict(x)[0].any():
    chart.add_rows(np.arange(num_out) / STEP, np.array(model.predict(x)[0]))


progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")