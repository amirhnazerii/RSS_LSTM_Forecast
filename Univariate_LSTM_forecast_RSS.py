import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


df_ini = pd.read_csv("C:/Users/anazeri/OneDrive - Clemson University/Research/Automotive_DrPisu/DynamicChannelSwitching/2022_Conf_paper/RSS_5G4G_Ad_Ad.csv").dropna()
data_ini= df_ini.to_numpy()
data_ini = data_ini.astype(float)
RSS_5G = data_ini[:,0]
RSS_4G = data_ini[:,1]

RSS_5G[RSS_5G <-90] = -90
RSS_5G[RSS_5G >-60] = -70

RSS_4G[RSS_4G <-60] = -60
RSS_4G[RSS_4G >-52] = -54

## Plot for paper
plot = False
if plot:
    plt.plot(RSS_5G, color = "red" ,label = "RSSI 5G")
    plt.xlabel('Time Index')
    plt.ylabel('RSSI (dBm)')
    plt.legend()
    plt.grid()
    plt.savefig('C:/Users/anazeri/OneDrive - Clemson University/Research/Automotive_DrPisu/DynamicChannelSwitching/2022_Conf_paper/RSSI 5G.png', dpi=400)
    plt.show()

    plt.plot(RSS_4G, color = "blue", label = "RSSI 4G")
    plt.xlabel('Time Index')
    plt.ylabel('RSSI (dBm)')
    plt.legend()
    plt.grid()
    plt.savefig('C:/Users/anazeri/OneDrive - Clemson University/Research/Automotive_DrPisu/DynamicChannelSwitching/2022_Conf_paper/RSSI 4G.png', dpi=400)



## Min-Max transformation

def Normalizer(unscaled_file, xmin, xmax, min, max):
    Norm_list = []
    """
    Output: Normalized batch of M-data : (M, N) where N is # of feauters, M = # of New_time_stamp
    """
    def MinMaxCal(X):

        """
        input: a single data point : 1D array : ["Load", "Minute",'hours', 'daylight', 'DayOfWeek', 'WeekDay'] for time t0
        output: Normalized signle data point: 1D array : (1, N)- N is # of feauters : 'Minute', 'hours', 'daylight', 'DayOfWeek', 'WeekDay'
        min = 0, max = 1, xmin = np.array([0, 0 , 0, 0, 0]), xmax = np.array([55, 23, 1, 6, 1])
        xmin, xmax are N-element arrayes for N features:  1D array
        """

        return ((X-xmin)/(xmax-xmin))*((max-min)+min)

    for i in range(len(unscaled_file)):
        Norm_list.append(MinMaxCal(unscaled_file[i,:]))
    Normalized_data = np.asarray(Norm_list)

    return Normalized_data

def DeNormalizer(scaled_file, xmin, xmax):

    DeNorm_list = []
    """
    Output: Normalized batch of M-data : (M, N) where N is # of feauters, M = # of New_time_stamp
    """
    def DeMinMaxCal(X):

        """
        input: a single data point : 1D array : ["Load", "Minute",'hours', 'daylight', 'DayOfWeek', 'WeekDay'] for time t0
        output: Normalized signle data point: 1D array : (1, N)- N is # of feauters : 'Minute', 'hours', 'daylight', 'DayOfWeek', 'WeekDay'
        min = 0, max = 1, xmin = np.array([0, 0 , 0, 0, 0]), xmax = np.array([55, 23, 1, 6, 1])
        xmin, xmax are N-element arrayes for N features:  1D array
        """

        return X*(xmax-xmin) +xmin

    for i in range(len(scaled_file)):
        DeNorm_list.append(DeMinMaxCal(scaled_file[i,:]))
    DeNormalized_data = np.asarray(DeNorm_list)

    return DeNormalized_data


shift = 0
df_size = 20000
RSS = RSS_4G
data_train_unscaled = RSS[shift : shift+df_size]
data_forecast_unscaled = RSS[     shift+df_size : shift+df_size+400]
data_unscaled = RSS[                      shift : shift+df_size+400]


min = 0
max = 1
x_min = np.array([RSS.min()])
x_max = np.array([RSS.max()])

testsize = 2000

data_scaled = Normalizer(RSS.reshape(-1,1), x_min, x_max, min, max)

data_train = data_scaled[shift : shift+df_size]
data_test = data_scaled[          shift+df_size : shift+df_size+testsize]
data_forecast = data_scaled[                     shift+df_size+testsize : shift+df_size+testsize+600]


#training/test split - sliding window

predict_time = 1 
unroll_length = 200

#Training data

def unroll(data, label ,sequence_length,target_length):
    dataset = []
    labels = []
    for index in range(len(data) - sequence_length-target_length):
        dataset.append(data[index : index + sequence_length])
        labels.append(label[        index + sequence_length : index + sequence_length + target_length])

    dataset_np = np.asarray(dataset)
    labels_np = np.asarray(labels)

    return dataset_np, labels_np

#Modified as LSTM standard input shape
future_len = 2
x_train, y_train = unroll(data_train, data_train, unroll_length, target_length = future_len)
x_test, y_test   = unroll(data_test,  data_test,  unroll_length, target_length = future_len)




 
# LSTM Model
num_features = 1  
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(100, input_shape= (unroll_length, num_features), return_sequences=False))
model.add(tf.keras.layers.LeakyReLU(alpha=0.5)) 
model.add(tf.keras.layers.Dropout(0.3)) 
# model.add(tf.keras.layers.LSTM(64, return_sequences=False))
# model.add(tf.keras.layers.Dropout(0.3)) 
# model.add(tf.keras.layers.LSTM(64, return_sequences=False))
model.add(tf.keras.layers.Dense(future_len))

opt = keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer=opt, loss='mse')

# Train the model
history = model.fit(x_train, y_train, batch_size=1024*1, epochs=40 , validation_split=0.2)

# plot training loss 
history_dict = history.history
history_dict.keys()
plt.figure(figsize = (12, 6))
plt.plot(model.history.history['loss'], label = 'loss')
plt.plot(model.history.history['val_loss'], label = 'val_loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.grid()


# predict x_test and plot 
xx = x_test[::future_len,:,:] # just batches with index multiple of future_len are selected.
predictions=model.predict( xx )
predictions_1 = predictions.reshape(-1,1)
x_test_forecasted = DeNormalizer(predictions_1, xmin =RSS.min() ,  xmax=RSS.max() )

y_test_modif = y_test[::future_len,:].reshape(-1,1)
y_test_modif = DeNormalizer(y_test_modif, xmin =RSS.min() , xmax=RSS.max()  ) #xmax=np.array([2])

# plt.figure(figsize=(8,5))
plt.plot(y_test_modif, color = 'blue', label = "Actual data")
plt.plot(x_test_forecasted, color = 'red', label = "LSTM forecast")
# plt.xlim([0, len(y_test_modif)])
plt.xlabel('Time Index')
plt.ylabel('RSSI (dBm)')
plt.legend()
plt.grid()
plt.savefig('C:/Users/anazeri/OneDrive - Clemson University/Research/Automotive_DrPisu/DynamicChannelSwitching/2022_Conf_paper/Univar_LSTM_forecast_RSSI_4G.png', dpi=400)
