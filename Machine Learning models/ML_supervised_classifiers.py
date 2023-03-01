import os
from collections import defaultdict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')

print('starting..')
dataset_root = '/path/to/'
train_file = os.path.join(dataset_root, 'iot_dataset.csv')
test_file = os.path.join(dataset_root, 'iot_dataset_very_test.csv')

header_names = ['Flow_ID', 'Src_IP', 'Src_Port','Dst_IP','Dst_Port','Protocol','Timestamp','Flow_Duration','Tot_Fwd_Pkts','Tot_Bwd_Pkts','TotLen_Fwd_Pkts','TotLen_Bwd_Pkts','Fwd_Pkt_Len_Max','Fwd_Pkt_Len_Min','Fwd_Pkt_Len_Mean','Fwd_Pkt_Len_Std','Bwd_Pkt_Len_Max','Bwd_Pkt_Len_Min','Bwd_Pkt_Len_Mean','Bwd_Pkt_Len_Std','Flow_Byts/s','Flow_Pkts/s','Flow_IAT_Mean','Flow_IAT_Std','Flow_IAT_Max','Flow_IAT_Min','Fwd_IAT_Tot','Fwd_IAT_Mean','Fwd_IAT_Std','Fwd_IAT_Max','Fwd_IAT_Min','Bwd_IAT_Tot','Bwd_IAT_Mean','Bwd_IAT_Std','Bwd_IAT_Max','Bwd_IAT_Min','Fwd_PSH_Flags','Bwd_PSH_Flags','Fwd_URG_Flags','Bwd_URG_Flags','Fwd_Header_Len','Bwd_Header_Len','Fwd_Pkts/s','Bwd_Pkts/s','Pkt_Len_Min','Pkt_Len_Max','Pkt_Len_Mean','Pkt_Len_Std','Pkt_Len_Var','FIN_Flag_Cnt','SYN_Flag_Cnt','RST_Flag_Cnt','PSH_Flag_Cnt','ACK_Flag_Cnt','URG_Flag_Cnt','CWE_Flag_Count','ECE_Flag_Cnt','Down/Up_Ratio','Pkt_Size_Avg','Fwd_Seg_Size_Avg','Bwd_Seg_Size_Avg','Fwd_Byts/b_Avg','Fwd_Pkts/b_Avg','Fwd_Blk_Rate_Avg','Bwd_Byts/b_Avg','Bwd_Pkts/b_Avg','Bwd_Blk_Rate_Avg','Subflow_Fwd_Pkts','Subflow_Fwd_Byts','Subflow_Bwd_Pkts','Subflow_Bwd_Byts','Init_Fwd_Win_Byts','Init_Bwd_Win_Byts','Fwd_Act_Data_Pkts','Fwd_Seg_Size_Min','Active_Mean','Active_Std','Active_Max','Active_Min','Idle_Mean','Idle_Std','Idle_Max','Idle_Min','Label','Cat','Sub_Cat']
col_names = np.array(header_names)

nominal_idx = [0,1,3,6,83,84,85]
numeric_idx = list(set(range(85)).difference(nominal_idx))

nominal_cols = col_names[nominal_idx].tolist()
numeric_cols = col_names[numeric_idx].tolist()


category = defaultdict(list)



train_df_old = pd.read_csv(train_file, names=header_names)
print('before')
print(train_df_old.isnull().sum().sum())
train_df_old.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df = train_df_old.dropna(axis=0)
print('after')
print(train_df.isnull().sum().sum())
train_df['attack_category'] = train_df['Cat']#.map(lambda x: attack_mapping[x])
train_df['attack_types'] = train_df['Sub_Cat']

    
test_df_old = pd.read_csv(test_file, names=header_names)
print('before')
print(test_df_old.isnull().sum().sum())
test_df_old.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df = test_df_old.dropna(axis=0)
print('after')
print(test_df.isnull().sum().sum())

test_df['attack_category'] = test_df['Cat']#.map(lambda x: attack_mapping[x])
test_df['attack_types'] = test_df['Sub_Cat']


#train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['Cat'].value_counts()
train_attack_types = train_df['Sub_Cat'].value_counts()
#test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['Cat'].value_counts()
print('plot for train attack cats')
train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=20)
print('plot for train attack types')
train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)
print('plot for test attack cats')
train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=20)
print('plot for test attack types')
train_attack_types.plot(kind='barh', figsize=(20,10), fontsize=20)


#DATA PREPARATION
print('starting data preparation!')
train_Y = train_df['attack_category']
train_x_raw = train_df.drop(['attack_category'], axis=1)
test_Y = test_df['attack_category']
test_x_raw = test_df.drop(['attack_category'], axis=1)

combined_df_raw = pd.concat([train_x_raw, test_x_raw])
combined_df = pd.get_dummies(combined_df_raw, columns=nominal_cols, drop_first=True)

train_x = combined_df[:len(train_x_raw)]
test_x = combined_df[len(train_x_raw):]


# Store dummy variable feature names
dummy_variables = list(set(train_x)-set(combined_df_raw))

train_x.describe()


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])

train_x.describe()


# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, zero_one_loss

classifier = DecisionTreeClassifier(random_state=17)
classifier.fit(train_x, train_Y)

pred_y = classifier.predict(test_x)

results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
print("Decision tree results:")

print(results)
print(error)

test_Y.value_counts().apply(lambda x: x/float(len(test_Y)))


#k-nearest neighbors classifier
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
classifier.fit(train_x, train_Y)

pred_y = classifier.predict(test_x)

results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
print("Knearest results:")
print(results)
print(error)


#Linear support vector classifier
from sklearn.svm import LinearSVC

classifier = LinearSVC()
classifier.fit(train_x, train_Y)

pred_y = classifier.predict(test_x)

results = confusion_matrix(test_Y, pred_y)
error = zero_one_loss(test_Y, pred_y)
print("Support vector results:")
print(results)
print(error)


print(pd.Series(train_Y).value_counts())

