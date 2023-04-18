import keras
from keras import regularizers, optimizers
from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
import numpy as np
from keras.layers import Conv1D, Flatten, Activation, SpatialDropout1D,add,GRU
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical


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

import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
import matplotlib.pyplot as plt
import hvplot.pandas

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, 
    roc_auc_score, roc_curve, auc,
    plot_confusion_matrix, plot_roc_curve
)

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, BatchNormalization 
from keras.optimizers import Adam
from keras.metrics import AUC


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

anomalies = df[df["loan_status"] == 1]
normal = df[df["loan_status"] == 0]
# 将指定行的''列值设置为一个常数，例如42

df2.loc[:, 'loan_status'] = 1


count = (df2['loan_status'] < 1).sum()

# 打印结果
#print(f"\nNumber of rows with 'loan_status' column value less than 1: {count}")

data_set = pd.concat([normal[:], anomalies,df2])
#print(anomalies.shape, normal.shape)
#df=data_set
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
#print(x_train)
#print(x_test)
y_train = x_train["loan_status"]
y_test = x_test["loan_status"]
x_train=x_train.drop("loan_status", axis=1)
x_test = x_test.drop("loan_status", axis=1)
print("----------------")
#print(x_train)
#print(x_test)
#print(y_train)
#print(x_train.head(10))
print("train samples:",x_train.shape[0])
print("test samples:",x_test.shape[0])



def print_score(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")

X_train = np.array(x_train).astype(np.float32)
X_test = np.array(x_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

#---ANN----#
def evaluate_nn(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
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
    plt.legend();

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


def nn_model(num_columns, num_labels, hidden_units, dropout_rates, learning_rate):
    inp = tf.keras.layers.Input(shape=(num_columns,1 ))
    #Series of temporal convolutional layers with dilations increasing by powers of 2.
    x = ResBlock(inp, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=4)
    x = GRU(64)(x)
    output = Dense(1, activation='sigmoid')(x)
  
    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=[AUC(name='AUC')])
    return model

num_columns = X_train.shape[1]
num_labels = 1
hidden_units = [150, 150, 150]
dropout_rates = [0.1, 0, 0.1, 0]
learning_rate = 1e-3


model = nn_model(
    num_columns=num_columns, 
    num_labels=num_labels,
    hidden_units=hidden_units,
    dropout_rates=dropout_rates,
    learning_rate=learning_rate
)
r = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    batch_size=32
)



plot_learning_evolution(r)

#y_train_pred = model.predict(X_train)
#evaluate_nn(y_train, y_train_pred.round(), train=True)

#y_test_pred = model.predict(X_test)
#evaluate_nn(y_test, y_test_pred.round(), train=False)

scores_dict = {
    'STCM': {
        'Train': roc_auc_score(y_train, model.predict(X_train)),
        'Test': roc_auc_score(y_test, model.predict(X_test)),
    },
}
#---ANN----#

#XGBoost Classifier
# param_grid = dict(
#     n_estimators=stats.randint(10, 500),
#     max_depth=stats.randint(1, 10),
#     learning_rate=stats.uniform(0, 1)
# )

xgb_clf = XGBClassifier(use_label_encoder=False)
# xgb_cv = RandomizedSearchCV(
#     xgb_clf, param_grid, cv=3, n_iter=60, 
#     scoring='roc_auc', n_jobs=-1, verbose=1
# )
# xgb_cv.fit(X_train, y_train)

# best_params = xgb_cv.best_params_
# best_params['tree_method'] = 'gpu_hist'
# # best_params = {'n_estimators': 50, 'tree_method': 'gpu_hist'}
# print(f"Best Parameters: {best_params}")

# xgb_clf = XGBClassifier(**best_params)
xgb_clf.fit(X_train, y_train)

y_train_pred = xgb_clf.predict(X_train)
y_test_pred = xgb_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)



scores_dict['XGBoost'] = {
        'Train': roc_auc_score(y_train, xgb_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, xgb_clf.predict(X_test)),
    }

#Random Forest Classifier
# param_grid = dict(
#     n_estimators=stats.randint(100, 1500),
#     max_depth=stats.randint(10, 100),
#     min_samples_split=stats.randint(1, 10),
#     min_samples_leaf=stats.randint(1, 10),
# )

rf_clf = RandomForestClassifier(n_estimators=100)
# rf_cv = RandomizedSearchCV(
#     rf_clf, param_grid, cv=3, n_iter=60, 
#     scoring='roc_auc', n_jobs=-1, verbose=1
# )
# rf_cv.fit(X_train, y_train)
# best_params = rf_cv.best_params_
# print(f"Best Parameters: {best_params}")
# rf_clf = RandomForestClassifier(**best_params)
rf_clf.fit(X_train, y_train)

y_train_pred = rf_clf.predict(X_train)
y_test_pred = rf_clf.predict(X_test)

print_score(y_train, y_train_pred, train=True)
print_score(y_test, y_test_pred, train=False)

scores_dict['Random Forest'] = {
        'Train': roc_auc_score(y_train, rf_clf.predict(X_train)),
        'Test': roc_auc_score(y_test, rf_clf.predict(X_test)),
    }

ml_models = {
    'Random Forest': rf_clf, 
    'XGBoost': xgb_clf, 
    'STCM': model
}

for model in ml_models:
    print(f"{model.upper():{30}} roc_auc_score: {roc_auc_score(y_test, ml_models[model].predict(X_test)):.3f}")

scores_df = pd.DataFrame(scores_dict)
p1=scores_df.hvplot.barh(
    width=500, height=400, 
    title="ROC Scores of ML Models", xlabel="ROC Scores", 
    alpha=0.4, legend='top'
)
hvplot.show(p1)