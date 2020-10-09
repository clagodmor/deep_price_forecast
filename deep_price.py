import os
from data.esios import *
from functools import reduce
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import streamlit as st
from time import gmtime, strftime

indicatorsDict = {'demand': 460,'price': 805,'wind':541,'solar':10034}

indicatorsItems = indicatorsDict.items()

start_date = datetime.date.today() - datetime.timedelta(days=7)
end_date = datetime.date.today()
now = strftime('T%H:%M:%S',gmtime())
start_ = start_date.strftime("%Y-%m-%d") + 'T00:00:00'
end_ = end_date.strftime("%Y-%m-%d") + now
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
date_time = pd.to_datetime(df.pop('Date'), format='%Y-%m-%d %H:%M:%S')
timestamp_s = date_time.map(datetime.datetime.timestamp)
day = 24*60*60
year = (365.2425)*day

df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# MODELO
features_considered = ['demand', 'solar', 'wind', 'price', 'Day sin', 'Day cos', 'Year sin', 'Year cos']
features = df[features_considered]
# features.index = df['Date']

dataset = features.values
data_mean = dataset.mean(axis=0)
data_std = dataset.std(axis=0)
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

past_history = 140
future_target = 12
STEP = 1
SINGLE_STEP = False
TRAIN_SPLIT = 0
BATCH_SIZE=16
x_data, y_data = multivariate_window(dataset=dataset, target=dataset[:, 3],
  start_index=TRAIN_SPLIT, end_index=None, history_size=past_history,
  target_size=future_target, step=STEP, single_step=SINGLE_STEP)
  
y_data = y_data.reshape((y_data.shape[0],y_data.shape[1], 1))

val_data = tf.data.Dataset.from_tensor_slices((x_data, y_data))
val_data = val_data.batch(BATCH_SIZE).repeat()



# Create a new model instance
model = keras.models.load_model('model/seq2seqcnn.h5')

# PLOT FUNCTION
def time_steps_creation(length):
  time_steps = []
  for i in range(-length, 0, 1):
    time_steps.append(i)
  return time_steps

st.write("""
# Spain Electrical Price forecast

The table below shows the prediction of the electrical price forecast. ements txt

It is calculating using a window of 5 days, and predicts 12 hours from the moment you run it. """)
for x, y in val_data.take(1):
    plt.figure(figsize=(12, 6))
    num_in = time_steps_creation(len(x[0]))
    num_out = len(y[0])

    plt.plot(num_in, np.array(x[0][:, 3]), label='History')
    # plt.plot(np.arange(num_out) / STEP, np.array(y[0]), 'bo',
    #          label='True Future')
    if model.predict(x)[0].any():
        plt.plot(np.arange(num_out) / STEP, np.array(model.predict(x)[0]), 'ro',
                 label='Predicted Future')
    plt.legend(loc='upper left')
    st.pyplot()

st.button("Re-run")

st.write("""
You can find how it works and all related information of this app at 
[GitHub](https://github.com/clagodmor/deep_price_forecast) """)
st.text("")
st.text("")
st.text("")
st.text("")
st.write("""
## About the author
 
**Clara Godoy** 
Analyst and Developer at [Maxam Int. Corp.](https://www.maxamcorp.com/)
 - Linkedin: [@claragodoy](https://www.linkedin.com/in/claragodoy/)
 - Twitter: [@clagodmor](https://twitter.com/clagodmor)""")
