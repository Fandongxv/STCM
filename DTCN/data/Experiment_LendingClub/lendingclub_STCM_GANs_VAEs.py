import keras
from keras import regularizers, optimizers
from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
import numpy as np
from keras.layers import Conv1D, Flatten, Activation, SpatialDropout1D,add,GRU, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
import os
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras.metrics import AUC
import tensorflow 
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)
#dataset
path = './data/08_20Q4/dataset_08_20.csv'
df = pd.read_csv(path, sep=",", index_col=None)
df=df.drop("mo_sin_old_il_acct",axis=1)
print(df.info())
print(pd.isnull(df).any())
# Standardize
#df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1))
#df['Time'] = StandardScaler().fit_transform(df['Time'].values.reshape(-1, 1))

path2='./data/reconstructed_lendingclub.csv'
df2 = pd.read_csv(path2, sep=",", index_col=None)
df2.loc[:, 'loan_status'] = 1
path3='./data/reconstructed_GANs_lendingclub.csv'
df3 = pd.read_csv(path3, sep=",", index_col=None)
df3.loc[:, 'loan_status'] = 1
path4='./data/reconstructed_VAEs_lendingclub.csv'
df4 = pd.read_csv(path4, sep=",", index_col=None)
df4.loc[:, 'loan_status'] = 1
anomalies = df[df["loan_status"] == 1]
normal = df[df["loan_status"] == 0]

data_set = pd.concat([normal[:], anomalies,df2,df3,df4])
#print(anomalies.shape, normal.shape)
df=data_set
'''
for f in range(0, 20):
    #间接处理：不改变原数据（对数组下标的处理）,随机20次
    normal = normal.iloc[np.random.permutation(len(normal))]
'''

#data_set = pd.concat([normal[:2000], anomalies])
#print(data_set)
#0.6训练集每次不一样，0.4测试集每次一样。
x_train, x_test = train_test_split(df, test_size = 0.4, random_state = 42)#test_set占0.4，random_state = 42重复执行的时候，测试集也是一样的。
print(x_test)
x_train = x_train.sort_index(axis=0)
x_test = x_test.sort_index(axis=0)
print("------------")
print(x_train)
print(x_test)
y_train = x_train["loan_status"]
y_test = x_test["loan_status"]
x_train=x_train.drop("loan_status", axis=1)
x_test = x_test.drop("loan_status", axis=1)
print("----------------")
print(x_train)
print(x_test)
print(y_train)
#print(x_train.head(10))
print("train samples:",x_train.shape[0])
print("test samples:",x_test.shape[0])

def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(
            classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    elif train == False:
        clf_report = pd.DataFrame(
            classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")


def evaluate_nn(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(
            classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
       # print("_______________________________________________")
       # print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

    elif train == False:
        clf_report = pd.DataFrame(
            classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
       # print("_______________________________________________")
       # print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")


def plot_learning_evolution(r):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(r.history['loss'], label='Loss')
    plt.plot(r.history['val_loss'], label='val_Loss')
    plt.title('Loss evolution during trainig')
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(r.history['AUC'], label='AUC')
    plt.plot(r.history['val_AUC'], label='val_AUC')
    plt.title('AUC score evolution during trainig')
    plt.legend()


'''-----------------TCN--------------------'''
# Residual block 残差块


def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(
        x)  # first convolution
    r = Conv1D(filters, kernel_size, padding='causal',
               dilation_rate=dilation_rate)(r)  # Second convolution

    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='causal')(
            x)  	# shortcut (shortcut)
    o = add([r, shortcut])
    # Activation function
    o = Activation('relu')(o)
    return o


# Sequence Model 时序模型
def TCN(train_x, train_y, valid_x, valid_y, test_x, test_y, classes, epoch):
    inputs = Input(shape=(61, 1))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=4)
    x = GRU(64)(x)
    #x = Dropout(0.3)(x)
    #x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    # View network structure 查看网络结构
    model.summary()
    # Compile model 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['mae', 'accuracy',tf.keras.metrics.AUC()])  # categorical_crossentropy mean_squared_error
    # Training model 训练模型
    history=model.fit(train_x, train_y, batch_size=500, epochs=epoch,
              verbose=1, validation_data=(valid_x, valid_y))
    
    plt.subplot(1,2,1)
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='lower right')

    plt.subplot(1,2,2)
    plt.plot(history.history['loss'],color='green')
    plt.plot(history.history['val_loss'],color='orange')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper right')
    plt.show()

    # Assessment model 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=1)
    print('test_loss:', pre[0], '- test_auc:', pre[1])
    return model


classes = 2
epoch = 1

y_train = keras.utils.to_categorical(y_train, 2)  # 将标签转为onehot。2为标签类别总数。
y_test = keras.utils.to_categorical(y_test, 2)
model = TCN(x_train, y_train, x_test,y_test, x_test, y_test, classes, epoch)
# y_train_pred = model.predict(x_train)
# print(y_train_pred)
# print(y_train_pred.round())
# evaluate_nn(y_train, y_train_pred.round(), train=True)

# auc1 = roc_auc_score(y_train, y_train_pred.round())
# print("AUC1: {:.2%}".format(auc1))
y_test_pred = model.predict(x_test)
print(classification_report(y_test_pred.round(), y_test))

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
y_pred2 = np.argmax(y_test_pred, axis=1)
y_test2 = np.argmax(y_test, axis=1)
visualize.draw_confusion_matrix(y_test2, y_pred2)
# x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)#

# x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

# input_shape = (x_train.shape[1], 1) #[31,1]
# y_train = keras.utils.to_categorical(y_train, 2)#将标签转为onehot。2为标签类别总数。
# y_test = keras.utils.to_categorical(y_test, 2)
# print(y_train)
# print("Shapes:\nx_train:%s\ny_train:%s\n" % (x_train.shape, y_train.shape))
# print("x_test:%s\ny_test:%s\n" % (x_test.shape, y_test.shape))
# print("input_shape:{}\n".format(input_shape))

# input_layer = Input(shape=(input_shape ))

# #Series of temporal convolutional layers with dilations increasing by powers of 2.
# conv_1 = Conv1D(filters=128, kernel_size=2, dilation_rate=1,
#                 padding='causal', strides=1,input_shape=input_shape,
#                 kernel_regularizer=regularizers.l2(0.01),
#                 activation='relu')(input_layer)

# #Dropout layer after each 1D-convolutional layer
# drop_1 = SpatialDropout1D(0.05)(conv_1)

# conv_2 = Conv1D(filters=128, kernel_size=2, dilation_rate=2,
#                 padding='causal',strides=1, kernel_regularizer=regularizers.l2(0.01),
#                 activation='relu')(drop_1)

# drop_2 = SpatialDropout1D(0.05)(conv_2)

# conv_3 = Conv1D(filters=128, kernel_size=2, dilation_rate=4,
#                 padding='causal', strides=1,kernel_regularizer=regularizers.l2(0.01),
#                 activation='relu')(drop_2)

# drop_3 = SpatialDropout1D(0.05)(conv_3)

# conv_4 = Conv1D(filters=128, kernel_size=2, dilation_rate=8,
#                 padding='causal', strides=1,kernel_regularizer=regularizers.l2(0.05),
#                 activation='relu')(drop_3)

# drop_4 = SpatialDropout1D(0.05)(conv_4)

# #Flatten layer to feed into the output layer
# flat = Flatten()(drop_4)

# output_layer = Dense(2, activation='softmax')(flat)

# TCN = Model(inputs=input_layer, outputs=output_layer)

# TCN.compile(loss='categorical_crossentropy',#categorical_crossentropy比mean好10%以上
#               optimizer=optimizers.Adam(lr=0.001),
#               metrics=['mae', 'accuracy'])

# checkpointer = ModelCheckpoint(filepath="model_TCN_creditcard.h5",
#                                verbose=0,
#                                save_best_only=True)

# #print(TCN.summary())

# TCN.fit(x_train, y_train,
#           batch_size=128,
#           epochs=1,
#           verbose=1,
#           validation_data=(x_test, y_test))

# score = TCN.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

# preds = TCN.predict(x_test)
# print(preds)
# y_pred = np.round(preds) #四舍五入
# print(y_pred)
# auc = roc_auc_score( y_test, y_pred)
# print("AUC: {:.2%}".format (auc))
# print(classification_report(y_test, y_pred))

class Visualization:
    labels = ["Normal", "Anomaly"]

    def draw_confusion_matrix(self, y, ypred):
        matrix = confusion_matrix(y, ypred)

        plt.figure(figsize=(10, 8))
        colors=[ "orange","green"]
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
y_pred2 = np.argmax(y_test_pred, axis=1)
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

#计算ROC-AUC
auc = roc_auc_score(y_test2, y_pred2)
# 输出结果
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1_score)
print("AUC:",auc)