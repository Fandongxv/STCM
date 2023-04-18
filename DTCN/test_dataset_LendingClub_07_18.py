import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import hvplot.pandas
import hvplot
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import hvplot.pandas
import hvplot
import keras
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import keras
from keras import regularizers, optimizers
from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence
import models.TCN
from keras.layers import Conv1D, Flatten, Activation, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import tensorflow 
import sys
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
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
from time import time
import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import keras
from keras import regularizers, optimizers
from keras import losses
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Input, Dropout, Embedding, LSTM
from keras.optimizers import RMSprop, Adam, Nadam
from keras.preprocessing import sequence

from keras.layers import Conv1D, Flatten, Activation, SpatialDropout1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.utils import to_categorical

import sklearn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


import tensorflow 
import sys
warnings.filterwarnings("ignore")
#dataset
path = './data/accepted_2007_to_2018Q4.csv'
df_orig = pd.read_csv(path, sep=",", index_col=None,low_memory=False)

#print(df_orig.head())

#loan_status是target
df = df_orig[(df_orig['loan_status']=='Fully Paid') | (df_orig['loan_status']=='Charged Off')]
#print(df.head())

#For training our pd model, we will only look at charged off and fully paid loans
df['loan_status_bin'] = df['loan_status'].map({'Charged Off': 1, 'Fully Paid': 0})

#Looking at the number of cutomers accepted in each grade, we can see that the grades with high number of customers are 'B' and 'C'
#(df['grade'].value_counts().sort_index()/len(df)).plot.bar()


#Before we can have a look at the distribution of the employment length we need to transform it into numerical values
def emp_to_num(term):
    if pd.isna(term):
        return None
    elif term[2]=='+':
        return 10
    elif term[0]=='<':
        return 0
    else:
        return int(term[0])

df['emp_length_num'] = df['emp_length'].apply(emp_to_num)
#(df['emp_length_num'].value_counts().sort_index()/len(df)).plot.bar()
#df.groupby('emp_length_num')['loan_status_bin'].mean().plot.bar()
#plt.show()

#Plotting the relation between employment length and default rate, we can see that there is no relation,i.e almost 2 to 9 have the same default rate, only <= 1 has a slightly higher and >10 has a slightly lower rate,so we will just divide the employment length into 2 variables 10+ and <=1 year
df['long_emp'] = df['emp_length'].apply(lambda x: 1*(x=='10+ years'))
df['short_emp'] = df['emp_length'].apply(lambda x: 1*(x=='1 year' or x=='< 1 year'))

#Let's have a look at the distribution of interest rate
(df['int_rate']/len(df)).plot.hist(bins=10)
#plt.show()

#Let's have a look at the distribution of annual income
#df[df['annual_inc']<200000]['annual_inc'].plot.hist(bins=20, rwidth=0.9)
df['annual_inc_log'] = df['annual_inc'].apply(np.log)
#plt.show()

#We can see that the most frequent purpose is debt consolidation
(df['purpose'].value_counts()/len(df)).plot.bar()


#Looking at the relationship between grade,default rate and interest rate
#We can see almost a linear relation between default rate and grade
df.groupby('grade')['loan_status_bin'].mean().plot.line()
#plt.show()

#We can also see an almost linear relation between interest rate and grade
df.groupby('grade')['int_rate'].mean().plot.line(color='blue')
#plt.show()

#As we can see there are only two possible values for the term on Lending Club, i.e. 36 months or 60 months.
(df['term'].value_counts()/len(df)).plot.bar(title='value counts')

#The accounts with a higher term have a significant higher default rate.
df.groupby('term')['loan_status_bin'].mean().plot.bar(title='default rate')
#plt.show()

#Doing the same with homeownership, the distribution among accounts is as follows ----mortgage,rent,own.
(df['home_ownership'].value_counts()/len(df)).plot.bar(title='value counts')
#plt.show()

#We can see that the default is higher in people who rent or own when compared to mortagage
df[(df['home_ownership']=='MORTGAGE') | (df['home_ownership']=='OWN')| (df['home_ownership']=='RENT')].groupby('home_ownership')['loan_status_bin'].mean().plot.bar(title='default rate')

#Taking a look at the fico score, we can see that most of out customers are in the FICO range 600-750
df['fico_range_high'].plot.hist(bins=20, title='FICO-Score', rwidth=0.9)
#plt.show()

#Also looking at the distribution for installment
df['installment'].plot.hist(bins=40, title='installment', rwidth=0.85)
#plt.show()

#We will use Linear Regression to calculate the linear function that maps from the default rate to the interest rate.
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(df.groupby('sub_grade')['loan_status_bin'].mean().values.reshape(-1,1), y=df.groupby('sub_grade')['int_rate'].mean())
plt.scatter(df.groupby('sub_grade')['loan_status_bin'].mean(), df.groupby('sub_grade')['int_rate'].mean())
plt.plot(df.groupby('sub_grade')['loan_status_bin'].mean(), lr.predict(df.groupby('sub_grade')['loan_status_bin'].mean().values.reshape(-1,1)))
plt.xlabel('default rate')
plt.ylabel('interest rate')
#plt.show()
#print('interest rate = ', lr.intercept_, '+', lr.coef_[0], '* default rate')
#From the above linear relation, we can see that when even in the same grade, if given a higher interest rate people are more likely to default when compared to a lower interest rate

#Reduce the dataset to the following columns that are known to investors before the loan is funded.
columns = ['loan_amnt', 'term', 'int_rate',
       'installment', 'grade', 'emp_length',
       'home_ownership', 'annual_inc_log', 'verification_status',
       'loan_status_bin', 'purpose',
       'addr_state', 'dti', 'delinq_2yrs',
       'fico_range_low', 'inq_last_6mths', 'open_acc',
       'pub_rec', 'revol_bal', 'revol_util', 'total_acc']
df = df[columns]

#Drop all rows that contain null-values.
df.dropna(inplace=True)

#Transform the grade into numerical values.
df['grade']=df['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})

#Get the dummy-variables for categorical features.
df_dummies = pd.get_dummies(df)

#We are going to drop all dummy-variables which contain not at least 1% ones. In this case we can simply look at the mean of the features because all non-dummy variables have means greater than 0.01.
drop_columns = df_dummies.columns[(df_dummies.mean()<0.01)]
df_dummies.drop(drop_columns, axis=1, inplace=True)

#Add the two different verification status variables that indicate verified to one variable.
df_dummies['verification_status_Verified_sum'] = df_dummies['verification_status_Source Verified']+df_dummies['verification_status_Verified']
df_dummies.drop(['verification_status_Source Verified', 'verification_status_Verified'], axis=1, inplace=True)

#print(df_dummies.head())
X=df_dummies
#Seperate features from targets.

# X = df_dummies.drop('loan_status_bin', axis=1)
# Y = df_dummies['loan_status_bin']


#print(X[loa].head())
print(X.shape[1])

anomalies = X[X["loan_status_bin"] == 1]
normal = X[X["loan_status_bin"] == 0]
#1比0为1：4，均衡。 
print(anomalies.shape, normal.shape)

k=X.pop("loan_status_bin")
X.insert(X.shape[1],"Class",k)
#print(X.columns)
#X.iloc[:,[0]]=StandardScaler().fit_transform(X.iloc[:,[0]].values.reshape(-1, 1))
#print(X.head())
#print(X.iloc[:,[0]])
# Standardize
#X=StandardScaler().fit_transform(X.values.reshape(-1, 1))
#print(X.shape)

# begin_time = time()                     # 标准化开始时间
# for i in range(X.shape[1]-1):
#     X.iloc[:,[i]]=StandardScaler().fit_transform(X.iloc[:,[i]].values.reshape(-1, 1))
# end_time = time()                      # 标准化结束时间
# total_time = end_time-begin_time       # 标准化耗时
# print('Standard Spending:',total_time,'s')
# #print(X.head())
# print(X.head(200))

anomalies = X[X["Class"] == 1]
normal = X[X["Class"] == 0]

print(anomalies.shape, normal.shape)


for f in range(0, 20):
    #间接处理：不改变原数据（对数组下标的处理）,随机20次
    normal = normal.iloc[np.random.permutation(len(normal))]
print("normal\n",normal)
data_set = pd.concat([normal[:7000000], anomalies])

train, test = train_test_split(data_set, test_size=0.4, random_state=42)

print(train.shape)
print(test.shape)
'''
#Removing Outliers
print(train[train['dti'] <= 50].shape)
print(train.shape)

print(train.shape)
#train = train[train['annual_inc'] <= 250000]
train = train[train['dti'] <= 50]
train = train[train['open_acc'] <= 40]
train = train[train['total_acc'] <= 80]
train = train[train['revol_util'] <= 120]
train = train[train['revol_bal'] <= 250000]
print(train.shape)
'''
#Normalizing the data
X_train, y_train = train.drop('Class', axis=1), train.Class
X_test, y_test = test.drop('Class', axis=1), test.Class

print(X_train.dtypes)

#归一化
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("X_train.shape" ,X_train.shape)

print("X_test.shape" ,X_test.shape)


#1比0为1：4，均衡。 
#print(anomalies.shape, normal.shape)
'''
print("X_train\n",x_train)
print("X_test\n",x_test)

print("Y_train\n",y_train)
print("Y_test\n",y_test)

'''
# print(x_train.head(10))
# print(x_train.shape[0])
# x_train = np.array(x_train).reshape(x_train.shape[0], x_train.shape[1], 1)#理解为1495个时刻，每个时刻31个数据。

# x_test = np.array(x_test).reshape(x_test.shape[0], x_test.shape[1], 1)

# input_shape = (x_train.shape[1], 1) #[31,1]
# y_train = keras.utils.to_categorical(y_train, 2)#将标签转为onehot。2为标签类别总数。
# y_test = keras.utils.to_categorical(y_test, 2)
# print(y_test[:500])
# print("Shapes:\nx_train:%s\ny_train:%s\n" % (x_train.shape, y_train.shape))
# print("x_test:%s\ny_test:%s\n" % (x_test.shape, y_test.shape))
# print("input_shape:{}\n".format(input_shape))
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


X_train = np.array(X_train).astype(np.float32)
X_test = np.array(X_test).astype(np.float32)
y_train = np.array(y_train).astype(np.float32)
y_test = np.array(y_test).astype(np.float32)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def evaluate_nn(true, pred, train=True):
    if train:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(true, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
       # print("_______________________________________________")
       # print(f"Confusion Matrix: \n {confusion_matrix(true, pred)}\n")
        
    elif train==False:
        clf_report = pd.DataFrame(classification_report(true, pred, output_dict=True))
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
    plt.legend();

# '''Artificial Neural Networks (ANNs)'''
# def nn_model(num_columns, num_labels, hidden_units, dropout_rates, learning_rate):
#      inp = tf.keras.layers.Input(shape=(num_columns, ))
#      x = BatchNormalization()(inp)
#      x = Dropout(dropout_rates[0])(x)
#      for i in range(len(hidden_units)):
#          x = Dense(hidden_units[i], activation='relu')(x)
#          x = BatchNormalization()(x)
#          x = Dropout(dropout_rates[i + 1])(x)
#      x = Dense(num_labels, activation='sigmoid')(x)
  
#      model = Model(inputs=inp, outputs=x)
#      model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=[AUC(name='AUC')])
#      return model

# num_columns = X_train.shape[1]
# num_labels = 1
# hidden_units = [150, 150, 150]
# dropout_rates = [0.1, 0, 0.1, 0]
# learning_rate = 1e-3


# model = nn_model(
#     num_columns=num_columns, 
#     num_labels=num_labels,
#     hidden_units=hidden_units,
#     dropout_rates=dropout_rates,
#     learning_rate=learning_rate
# )
# r = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=10,
#     batch_size=32
# )

# plot_learning_evolution(r)
# plt.show()

# y_train_pred = model.predict(X_train)
# evaluate_nn(y_train, y_train_pred.round(), train=True)

# y_test_pred = model.predict(X_test)
# evaluate_nn(y_test, y_test_pred.round(), train=False)

''''-----------------TCN--------------------'''
# Residual block 残差块
def ResBlock(x, filters, kernel_size, dilation_rate):
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate, activation='relu')(
        x)  # first convolution
    r = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate)(r)  # Second convolution


    if x.shape[-1] == filters:
        shortcut = x
    else:
        shortcut = Conv1D(filters, kernel_size, padding='same')(x)  	# shortcut (shortcut)
    o = add([r, shortcut])
    # Activation function
    o = Activation('relu')(o)  
    return o


# Sequence Model 时序模型
def TCN(train_x, train_y, valid_x, valid_y, test_x, test_y, classes, epoch):
    inputs = Input(shape=(68, 1))
    x = ResBlock(inputs, filters=32, kernel_size=3, dilation_rate=1)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=2)
    x = ResBlock(x, filters=32, kernel_size=3, dilation_rate=4)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=x)
    # View network structure 查看网络结构
    model.summary()
    # Compile model 编译模型
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[AUC(name='AUC')])#categorical_crossentropy
    # Training model 训练模型
    model.fit(train_x, train_y, batch_size=500, epochs=epoch, verbose=1, validation_data=(valid_x, valid_y))
    # Assessment model 评估模型
    pre = model.evaluate(test_x, test_y, batch_size=500, verbose=1)
    print('test_loss:', pre[0], '- test_auc:', pre[1])
    return model
classes = 2
epoch = 1

y_train = keras.utils.to_categorical(y_train, 2)#将标签转为onehot。2为标签类别总数。
y_test = keras.utils.to_categorical(y_test, 2)
model=TCN(X_train, y_train, X_test, y_test, X_test, y_test, classes, epoch)
y_train_pred = model.predict(X_train)
print(y_train_pred)
print(y_train_pred.round())
evaluate_nn(y_train, y_train_pred.round(), train=True)

y_test_pred = model.predict(X_test)
evaluate_nn(y_test, y_test_pred.round(), train=False)
y_test_pred=y_test_pred.round() #为了下面的auc，让数据从连续变成离散
auc = roc_auc_score( y_test_pred, y_test)
print("AUC: {:.2%}".format (auc))
print(classification_report(y_test, y_test_pred))
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