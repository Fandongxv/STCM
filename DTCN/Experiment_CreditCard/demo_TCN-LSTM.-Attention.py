import keras
from keras import regularizers, optimizers
from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM,GRU,Bidirectional,Multiply,Permute
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
from keras.layers import Conv1D, Flatten, Activation, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import tensorflow as tf
from tensorflow import keras


import tensorflow 
import sys

#dataset
path = './data/creditcard.csv'
df = pd.read_csv(path, sep=",", index_col=None)


# Standardize
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

anomalies = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

#print(anomalies.shape, normal.shape)


for f in range(0, 20):
    #间接处理：不改变原数据（对数组下标的处理）,随机20次
    normal = normal.iloc[np.random.permutation(len(normal))]

data_set = pd.concat([normal[:2000], anomalies])
#print(data_set)
#0.6训练集每次不一样，0.4测试集每次一样。
x_train, x_test = train_test_split(data_set, test_size = 0.4, random_state = 42)#test_set占0.4，random_state = 42重复执行的时候，测试集也是一样的。
print(x_test)
x_train = x_train.sort_values(by=['Time'])
x_test = x_test.sort_values(by=['Time'])
#print(x_train)
#print(x_test)
y_train = x_train["Class"]
y_test = x_test["Class"]
#print(y_train)

print(x_train.head(10))
print(x_train.shape[0])
x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)#理解为1495个时刻，每个时刻31个数据。

x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

input_shape = (x_train.shape[1], 1) #[31,1]
y_train = keras.utils.to_categorical(y_train, 2)#将标签转为onehot。2为标签类别总数。
y_test = keras.utils.to_categorical(y_test, 2)
print(y_train)
print("Shapes:\nx_train:%s\ny_train:%s\n" % (x_train.shape, y_train.shape))
print("x_test:%s\ny_test:%s\n" % (x_test.shape, y_test.shape))
print("input_shape:{}\n".format(input_shape))

# 注意力机制
def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = inputs
    a = Dense(input_dim, activation='softmax')(a)
    # 根据给定的模式(dim)置换输入的维度  例如(2,1)即置换输入的第1和第2个维度
    a_probs = Permute((1, 2), name='attention_vec')(a)
    # Layer that multiplies (element-wise) a list of inputs.
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

input_layer = Input(shape=(input_shape ))

#Series of temporal convolutional layers with dilations increasing by powers of 2.
conv_1 = Conv1D(filters=32, kernel_size=3, dilation_rate=1,
                padding='causal', strides=1,input_shape=input_shape,
                kernel_regularizer=regularizers.l2(0.01),
                activation='relu')(input_layer)

#Dropout layer after each 1D-convolutional layer
drop_1 = SpatialDropout1D(0.3)(conv_1)

conv_2 = Conv1D(filters=32, kernel_size=3, dilation_rate=2,
                padding='causal',strides=1, kernel_regularizer=regularizers.l2(0.01),
                activation='relu')(drop_1)

drop_2 = SpatialDropout1D(0.3)(conv_2)

conv_3 = Conv1D(filters=32, kernel_size=3, dilation_rate=4,
                padding='causal', strides=1,kernel_regularizer=regularizers.l2(0.01),
                activation='relu')(drop_2)

drop_3 = SpatialDropout1D(0.3)(conv_3)

conv_4 = Conv1D(filters=32, kernel_size=3, dilation_rate=8,
                padding='causal', strides=1,kernel_regularizer=regularizers.l2(0.05),
                activation='relu')(drop_3)

drop_4 = SpatialDropout1D(0.3)(conv_4)
lstm_out = Bidirectional(LSTM(64, return_sequences=True))(drop_4)
#Flatten layer to feed into the output layer
#flat = Flatten()(drop_5)
lstm_out = Dropout(0.3)(lstm_out)
attention_mul = attention_3d_block(lstm_out)
attention_mul = Flatten()(attention_mul)
output_layer = Dense(2, activation='softmax')(attention_mul)

TCN = Model(inputs=input_layer, outputs=output_layer)

TCN.compile(loss='mean_squared_error',
              optimizer=optimizers.Adam(lr=0.002),
              metrics=['mae', 'accuracy',tf.keras.metrics.AUC()])

checkpointer = ModelCheckpoint(filepath="model_TCN_creditcard.h5",
                               verbose=0,
                               save_best_only=True)

#print(TCN.summary())

history=TCN.fit(x_train, y_train,
          batch_size=500,
          epochs=100,
          verbose=1,
          validation_data=(x_test, y_test))
# plt.subplot(1,2,1)
# plt.plot(history.history['auc'])
# plt.plot(history.history['val_auc'])
# plt.title('AUC')
# plt.ylabel('AUC')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='lower right')

# plt.subplot(1,2,2)
# plt.plot(history.history['loss'],color='green')
# plt.plot(history.history['val_loss'],color='orange')
# plt.title('Model Loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['train', 'validation'], loc='upper right')
# plt.show()

score = TCN.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

preds = TCN.predict(x_test)
print(preds)
y_pred = np.round(preds) #四舍五入
print(y_pred)
auc = roc_auc_score( y_test, y_pred)
print("AUC: {:.2%}".format (auc))
print(classification_report(y_test, y_pred))


class Visualization:
    labels = ["Normal", "Anomaly"]

    def draw_confusion_matrix(self, y, ypred):
        matrix = confusion_matrix(y, ypred)

        plt.figure(figsize=(10, 8))
        colors=[ "darkgreen","lightblue"]
        sns.heatmap(matrix, xticklabels=self.labels, yticklabels=self.labels, cmap=colors, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()


    def draw_anomaly(self, y, error, threshold):
        groupsDF = pd.DataFrame({'error': error,
                                 'true': y}).groupby('true')

        figure, axes = plt.subplots(figsize=(12, 8))

        for name, group in groupsDF:
            axes.plot(group.index, group.error, marker='x' if name == 1 else 'o', linestyle='',
                    color='r' if name == 1 else 'g', label="Anomaly" if name == 1 else "Normal")

        axes.hlines(threshold, axes.get_xlim()[0], axes.get_xlim()[1], colors="b", zorder=100, label='Threshold')
        axes.legend()
        
        plt.title("Anomalies")
        plt.ylabel("Error")
        plt.xlabel("Data")
        plt.show()

    def draw_error(self, error, threshold):
            plt.plot(error, marker='o', ms=3.5, linestyle='',
                     label='Point')

            plt.hlines(threshold, xmin=0, xmax=len(error)-1, colors="b", zorder=100, label='Threshold')
            plt.legend()
            plt.title("Reconstruction error")
            plt.ylabel("Error")
            plt.xlabel("Data")
            plt.show()

visualize = Visualization()
y_pred2 = np.argmax(y_pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)
visualize.draw_confusion_matrix(y_test2, y_pred2)

# 计算Accuracy
accuracy = accuracy_score(y_test2, y_pred2)

# 计算Precision
precision = precision_score(y_test2, y_pred2)

# 计算Recall
recall = recall_score(y_test2, y_pred2)

# 计算F1-score
f1_score = f1_score(y_test2, y_pred2)

# 输出结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)