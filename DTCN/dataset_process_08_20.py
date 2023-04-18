import os
import time
import pandas as pd
from zipfile import ZipFile

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 200)

import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter

data_path='./data/08_18Q4'
data = pd.read_csv('./data/Loan_status_2007-2020Q3.csv')
print(len(data))
print(data.shape)
data=data.reset_index(drop=True)

print(data.groupby('loan_status').size())


loan_status_dict = {"Fully Paid": 0,
                    "Charged Off": 1,
                    "Late (31-120 days)": 1,
                    "Late (16-30 days)": 1,
                    "Default": 1,
                    "Current": -1,
                    "In Grace Period": -1,
                    "Issued": -1,
                    "Does not meet the credit policy. Status:Charged Off": -1,
                    "Does not meet the credit policy. Status:Fully Paid": -1
                    }
data["loan_status"] = data["loan_status"].map(loan_status_dict)

data = data[data["loan_status"]!=-1]
#print(data.groupby('loan_status').size())

total_misval = data.isna().sum().sort_values(ascending=False) 
total_misval = total_misval[total_misval != 0] 
per_misval = total_misval / total_misval.max()  


def draw_per_misval():
    f, ax = plt.subplots(figsize=(10, 10),dpi=100)
    sns.set_style("whitegrid")
    
    
    sns.barplot(x=per_misval[per_misval>0.1]*100, 
                y=per_misval[per_misval>0.1].index, 
                ax=ax,
                palette="GnBu_r")
    
    ax.xaxis.set_major_formatter(FormatStrFormatter("%2.f%%")) 
    ax.set_title("missing value")
    plt.show()
#draw_per_misval()


def drop_misval_ft(data, threshold, per_misval):
    misval_ft = per_misval[per_misval > threshold].index
    data.drop(misval_ft, axis=1, inplace=True)
    print("delete {} fealtures".format(len(misval_ft)))
    return data


def drop_samples(data, threshold, per_misval):
    features = per_misval[per_misval < threshold].index

    print("{} samples once".format(data.shape[0]))
    data.dropna(subset=features, inplace=True) 
    print("deleted! {}samples now!".format(data.shape[0]))
    return data

data = drop_misval_ft(data, 0.15, per_misval)
data = drop_samples(data, 0.05, per_misval)

data["mths_since_recent_inq"].hist()
#plt.show()
data["mths_since_recent_inq"].fillna(0,inplace=True) 
data.drop("emp_title",axis=1,inplace=True)

emp_length_dict = {"10+ years": 10, "2 years": 2, "< 1 year": 0.5, "3 years": 3, "1 year": 1, "5 years": 5,
                   "4 years": 4, "6 years": 6, "7 years": 7, "8 years": 8, "9 years": 9}
data["emp_length"] = data["emp_length"].map(emp_length_dict)
data["emp_length"].fillna(value=0, inplace=True)
print(data.groupby("emp_length").size())


def drop_high_freq_features(df, freq_limit):
    high_freq_features = []
    for feature in df.columns:
        n = df.shape[0] 
        most_ft_val = df[feature].value_counts().max() 
        per = most_ft_val/n 
        if per >freq_limit:
            high_freq_features.append(feature)
    
    df.drop(high_freq_features,axis=1,inplace=True) 
    print("delete {} features".format(len(high_freq_features)))
    print(" remain {} features ".format(df.shape[1]))
    return df
data = drop_high_freq_features(data, freq_limit=0.95)

data.issue_d = data.issue_d.apply(lambda x:x[-4:])



print(data.info())

dorp_features = [
                "id",
                "funded_amnt",
                "funded_amnt_inv",
                
                "url",
                "zip_code",
                "addr_state",
                "earliest_cr_line",  
                "total_pymnt",
                "total_pymnt_inv",
                "total_rec_prncp",
                "total_rec_int",
                "total_rec_late_fee",
                
                "last_pymnt_amnt",  
                "last_pymnt_d",  
                "last_credit_pull_d",  
                "loan_status"
                
                 ]
labels = data['loan_status'].copy()

data.drop(dorp_features, axis=1, inplace=True)




def per2float(df):
   
    for feature in df.columns:
        if data[feature].dtype != 'O':
            continue
        if "%" in str(df[feature].iloc[0]):
            
            print(feature)
            df[feature] = df[feature].apply(lambda x: float(x.strip("%"))) / 100
    return data

data = per2float(data)


num_features = data.select_dtypes('number').columns
data[num_features] = (data[num_features] - data[num_features].mean()) / data[num_features].std()


from sklearn.preprocessing import LabelEncoder
category_features = data.select_dtypes('object').columns
data[category_features] = data[category_features].apply(lambda x: LabelEncoder().fit_transform(x))

data = pd.concat([data, labels], axis=1)

print(data.head())

#data.to_csv(os.path.join(data_path, 'dataset_08_18.csv.zip'), index=None, compression='zip')
print(pd.isnull(data).any())
print(pd.value_counts(data["loan_status"]==1))
print(pd.value_counts(data["loan_status"]==0))