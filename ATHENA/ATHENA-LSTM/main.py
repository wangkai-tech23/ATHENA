import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model
from keras import regularizers
from sklearn.metrics import recall_score, confusion_matrix, roc_curve, auc, f1_score, precision_score


def create_dataset(data, timesteps):
    X = []
    for i in range(len(data) - timesteps):
        X.append(data[i:i+timesteps])
    return np.array(X)

def autoencoder_model(X):
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    L1 = LSTM(16, activation='relu', return_sequences=True, 
              kernel_regularizer=regularizers.l2(0.00))(inputs)
    L2 = LSTM(16, activation='relu', return_sequences=False)(L1)
    L3 = RepeatVector(X.shape[1])(L2)
    L4 = LSTM(16, activation='relu', return_sequences=True)(L3)
    L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
    output = TimeDistributed(Dense(1))(L5)  # 修改输出维度为1
    model = Model(inputs=inputs, outputs=output)
    return model

data = pd.read_csv('data/all/train/0D0.csv')
valid_data = pd.read_csv('data/all/validation/0D0_1.csv')

time_series = data['Time'].values.reshape(-1, 1)

time_intervals = np.diff(data['Time'])

scaler = MinMaxScaler()
normalized_intervals = time_intervals.reshape(-1, 1)


timesteps = 1
X_train = create_dataset(normalized_intervals, timesteps)

model = autoencoder_model(X_train)
model.compile(optimizer='adam', loss='mae')

model.fit(X_train, X_train, epochs=10, batch_size=32, validation_split=0.1)
predict_y = model.predict(X_train)[:, 0, :]
train_error = np.abs(predict_y - X_train[:, 0, :])

valid_intervals = valid_data['Time'].diff().dropna()
case_numbers = valid_data.index[1:].values
valid_intervals

abnormals = []
is_attack_predicted = []
tmp_len = 0
for i in range(len(valid_intervals)):
    # thr = predict_y.reshape(-1)[i] - 2 * np.mean(train_error)
    thr = predict_y.reshape(-1)[i] - 2 * np.std(predict_y.reshape(-1))
    value = valid_intervals.values[i]
    if tmp_len != 0:
        value += tmp_len
        tmp_len = 0
    if value < thr:
        abnormals.append(case_numbers[i])
        is_attack_predicted.append(1)
        tmp_len = valid_intervals.values[i]
    else:
        is_attack_predicted.append(0)
        
valid_data['is_attack'] = valid_data['Label'].apply(lambda x: 0 if x == 'Normal' else 1)
is_attack_actual = valid_data['is_attack'].values[1:] # 对其处理数据时的索引差异

conf_matrix = confusion_matrix(is_attack_actual, is_attack_predicted)
print("混淆矩阵:")
print(conf_matrix)

f1 = f1_score(is_attack_actual, is_attack_predicted)
print(f"F1 分数: {f1}")

recall = recall_score(is_attack_actual, is_attack_predicted)
print(f"召回率: {recall}")

accuracy = precision_score(is_attack_actual, is_attack_predicted)
print(f"准确率: {accuracy}")
