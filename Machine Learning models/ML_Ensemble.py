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
test_file = os.path.join(dataset_root, 'iot_dataset_test.csv')

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

    
test_df_old = pd.read_csv(test_file, names=header_names)
print('before')
print(test_df_old.isnull().sum().sum())
test_df_old.replace([np.inf, -np.inf], np.nan, inplace=True)
test_df = test_df_old.dropna(axis=0)
print('after')
print(test_df.isnull().sum().sum())

test_df['attack_category'] = test_df['Cat']#.map(lambda x: attack_mapping[x])


#train_attack_types = train_df['attack_type'].value_counts()
train_attack_cats = train_df['Cat'].value_counts()

#test_attack_types = test_df['attack_type'].value_counts()
test_attack_cats = test_df['Cat'].value_counts()

train_attack_cats.plot(kind='barh', figsize=(20,10), fontsize=20)



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


#dummy variable
dummy_variables = list(set(train_x)-set(combined_df_raw))

train_x.describe()


from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler().fit(train_x[numeric_cols])

train_x[numeric_cols] = \
    standard_scaler.transform(train_x[numeric_cols])

test_x[numeric_cols] = \
    standard_scaler.transform(test_x[numeric_cols])

train_x.describe()

train_Y_bin = train_Y.apply(lambda x: 0 if x is 'Normal' else 1)
test_Y_bin = test_Y.apply(lambda x: 0 if x is 'Normal' else 1)



#oversampling - undersampling

test_Y.value_counts().apply(lambda x: x/float(len(test_Y)))
train_Y.value_counts().apply(lambda x: x/float(len(train_Y)))
print(pd.Series(train_Y).value_counts())

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=0)
train_x_sm, train_Y_sm = sm.fit_resample(train_x, train_Y)
print(pd.Series(train_Y_sm).value_counts())


####
from imblearn.under_sampling import RandomUnderSampler

mean_class_size = int(pd.Series(train_Y).value_counts().sum()/5)

ratio = {'Mirai': mean_class_size,
         'Scan': mean_class_size,
         'DoS': mean_class_size,
         'Normal': mean_class_size,
         'MITM ARP Spoofing': mean_class_size}

rus = RandomUnderSampler(sampling_strategy=ratio, random_state=0, replacement=True)
train_x_rus, train_Y_rus = rus.fit_resample(train_x_sm, train_Y_sm)
print(pd.Series(train_Y_rus).value_counts())

#Dataset visualization using PCA for two dimensions

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
train_x_pca_cont = pca.fit_transform(train_x[numeric_cols])




from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=17).fit(train_x[numeric_cols])
kmeans_y = kmeans.labels_


plt.figure(figsize=(15,10))
colors = ['navy', 'turquoise', 'darkorange', 'red', 'purple']

for color, cat in zip(colors, range(5)):
    plt.scatter(train_x_pca_cont[kmeans_y==cat, 0],
                train_x_pca_cont[kmeans_y==cat, 1],
                color=color, alpha=.8, lw=2, label=cat)
plt.legend(loc='best', shadow=False, scatterpoints=1)

plt.show()

print('Total number of features: {}'.format(len(train_x.columns)))
print('Total number of continuous features: {}'.format(len(train_x[numeric_cols].columns)))


from sklearn.metrics import completeness_score,\
    homogeneity_score, v_measure_score, accuracy_score




##################### AR ##################################################
averages = train_df.loc[:, numeric_cols].mean()

averages_per_class = train_df[numeric_cols+['attack_category']].groupby('attack_category').mean()

AR = {}
for col in numeric_cols:
    AR[col] = max(averages_per_class[col])/averages[col]

print(AR)

def binary_AR(df, col):
    series_zero = train_df[train_df[col] == 0].groupby('attack_category').size()
    series_one = train_df[train_df[col] == 1].groupby('attack_category').size()
    return max(series_one/series_zero)

labels2 = ['Normal', 'Attack']
labels5 = ['Normal', 'Mirai', 'Scan', 'DoS', 'MITM ARP Spoofing']


train_df['labels2'] = train_df.apply(lambda x: 'Normal' if 'Normal' in x['Cat'] else 'Attack', axis=1)
test_df['labels2'] = test_df.apply(lambda x: 'Normal' if 'Normal' in x['Cat'] else 'Attack', axis=1)

combined_df = pd.concat([train_df, test_df])
original_cols = combined_df.columns

combined_df = pd.get_dummies(combined_df, columns=nominal_cols, drop_first=True)

added_cols = set(combined_df.columns) - set(original_cols)
added_cols= list(added_cols)

combined_df.attack_category = pd.Categorical(combined_df.attack_category)
combined_df.labels2 = pd.Categorical(combined_df.labels2)

combined_df['labels5'] = combined_df['attack_category'].cat.codes
combined_df['labels2'] = combined_df['labels2'].cat.codes

train_df = combined_df[:len(train_df)]
test_df = combined_df[len(train_df):]




import operator
AR = dict((k, v) for k,v in AR.items() if not np.isnan(v))
sorted_AR = sorted(AR.items(), key=lambda x:x[1], reverse=True)

#print(sorted_AR)


features_to_use = []
for x,y in sorted_AR:
    if y >= 0.01:
        features_to_use.append(x)
        
print(features_to_use)
train_df_trimmed = train_df[features_to_use]
test_df_trimmed = test_df[features_to_use]

numeric_cols_to_use = list(set(numeric_cols).intersection(features_to_use))

#Rescaling
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()

train_df_trimmed[numeric_cols_to_use] = standard_scaler.fit_transform(train_df_trimmed[numeric_cols_to_use])
test_df_trimmed[numeric_cols_to_use] = standard_scaler.transform(test_df_trimmed[numeric_cols_to_use])


###########################################################################

#train_Y_bin = train_Y.apply(lambda x: 'Normal' if x == 'Normal' else 'attack')


kmeans = KMeans(n_clusters=5, random_state=17)
kmeans.fit(train_df_trimmed[numeric_cols_to_use])
kmeans_train_y = kmeans.labels_

pd.crosstab(kmeans_train_y, train_Y_bin)





train_df['kmeans_y'] = kmeans_train_y
train_df_trimmed['kmeans_y'] = kmeans_train_y

kmeans_test_y = kmeans.predict(test_df_trimmed[numeric_cols_to_use])
test_df['kmeans_y'] = kmeans_test_y


pca8 = PCA(n_components=2)
train_df_trimmed_pca8 = pca8.fit_transform(train_df_trimmed)

plt.figure(figsize=(15,10))

colors8 = ['navy', 'turquoise', 'darkorange', 'red', 'purple', 'green', 'magenta', 'black']
labels8 = [0,1,2,3,4,5,6,7]

for color, cat in zip(colors8, labels8):
    plt.scatter(train_df_trimmed_pca8[train_df.kmeans_y==cat, 0], train_df_trimmed_pca8[train_df.kmeans_y==cat, 1],
                color=color, alpha=.8, lw=2, label=cat)


pd.crosstab(test_df.kmeans_y, test_df.labels5)




#Ensemble
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, zero_one_loss




train_y0 = train_df[train_df.kmeans_y==0]
test_y0 = test_df[test_df.kmeans_y==0]
rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=17).fit(train_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1), train_y0['labels5'])
pred_y0 = rfc.predict(test_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1))
print("cluster {} score is {}, {}".format(0, accuracy_score(pred_y0, test_y0['labels5']), accuracy_score(pred_y0, test_y0['labels5'], normalize=False)))

print(confusion_matrix(test_y0['labels5'], pred_y0))



train_y0 = train_df[train_df.kmeans_y==1]
test_y0 = test_df[test_df.kmeans_y==1]
rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=17).fit(train_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1), train_y0['labels5'])
pred_y1 = rfc.predict(test_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1))
print("cluster {} score is {}, {}".format(1, accuracy_score(pred_y1, test_y0['labels5']), accuracy_score(pred_y1, test_y0['labels5'], normalize=False)))

print(confusion_matrix(test_y0['labels5'], pred_y1))




train_y0 = train_df[train_df.kmeans_y==2]
test_y0 = test_df[test_df.kmeans_y==2]
rfc = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=17).fit(train_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1), train_y0['labels5'])
pred_y2 = rfc.predict(test_y0.drop(['labels2', 'labels5', 'kmeans_y', 'attack_category'], axis=1))
print("cluster {} score is {}, {}".format(2, accuracy_score(pred_y2, test_y0['labels5']), accuracy_score(pred_y2, test_y0['labels5'], normalize=False)))

print(confusion_matrix(test_y0['labels5'], pred_y2))


print(accuracy_score(test_df[test_df.kmeans_y==3]['labels5'], np.zeros(len(test_df[test_df.kmeans_y==3]))))
print(confusion_matrix(test_df[test_df.kmeans_y==3]['labels5'], np.zeros(len(test_df[test_df.kmeans_y==3]))))
pred_y3 = np.zeros(len(test_df[test_df.kmeans_y==3]))

print(accuracy_score(test_df[test_df.kmeans_y==4]['labels5'], np.zeros(len(test_df[test_df.kmeans_y==4]))))
print(confusion_matrix(test_df[test_df.kmeans_y==4]['labels5'], np.zeros(len(test_df[test_df.kmeans_y==4]))))
pred_y4 = np.zeros(len(test_df[test_df.kmeans_y==4]))



# combined results:
num_samples = 5262
false_pos = 9
false_neg = 4245

print('True positive %: {}'.format(1-(false_pos/num_samples)))
print('True negative %: {}'.format(1-(false_neg/num_samples)))

