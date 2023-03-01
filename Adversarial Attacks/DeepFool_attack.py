
from cleverhans.utils_tf import model_train , model_eval , batch_eval
from cleverhans.attacks_tf import jacobian_graph
from cleverhans.attacks_tf import  fgsm, deepfool_attack, deepfool_batch
from cleverhans.utils_keras import KerasModelWrapper
from cleverhans.attacks import FastGradientMethod, MadryEtAl, DeepFool, CarliniWagnerL2
from art.attacks import evasion
from art.attacks.evasion import DeepFool, FastGradientMethod, CarliniL2Method, BoundaryAttack,AutoAttack
from art.estimators.classification import KerasClassifier, TensorFlowClassifier
from art.attacks.poisoning import PoisoningAttackBackdoor

from cleverhans.utils import other_classes
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , roc_curve , auc , f1_score
from sklearn.preprocessing import LabelEncoder , MinMaxScaler
from tensorflow.python.platform import flags
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

from tensorflow.python.keras.optimizer_v2.adam import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, normalized_mutual_info_score

from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
import logging
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib . pyplot as plt
plt.style.use('bmh')

adam = Adam()





logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("[%(levelname)s] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)



names = ['Flow_ID', 'Src_IP', 'Src_Port', 'Dst_IP', 'Dst_Port', 'Protocol', 'Fwd_IAT_Tot', 'Bwd_IAT_Tot', 'Fwd_PSH_Flags', 'Bwd_PSH_Flags', 'Fwd_URG_Flags', 'Bwd_URG_Flags', 'Fwd_Header_Len', 'Bwd_Header_Len', 'Subflow_Fwd_Pkts', 'Subflow_Fwd_Byts', 'Subflow_Bwd_Pkts', 'Subflow_Bwd_Byts', 'Init_Fwd_Win_Byts', 'Init_Bwd_Win_Byts', 'Fwd_Act_Data_Pkts', 'Fwd_Seg_Size_Min', 'Active_Mean', 'Active_Std', 'Active_Max', 'Active_Min', 'Idle_Mean', 'Idle_Std', 'Idle_Max', 'Idle_Min', 'Label', 'Cat', 'Sub_Cat']


train_df_old = pd.read_csv('iot_dataset.csv', names=names , header=None)
test_df_old = pd.read_csv('iot_dataset_test.csv', names=names , header=None)

print('before')
print(train_df_old.isnull().sum().sum())
train_df_old.replace([np.inf, -np.inf], np.nan, inplace=True)
df = train_df_old.dropna(axis=0)
print('after')
print(df.isnull().sum().sum())


print('before')
print(test_df_old.isnull().sum().sum())
test_df_old.replace([np.inf, -np.inf], np.nan, inplace=True)
dft = test_df_old.dropna(axis=0)
print('after')
print(dft.isnull().sum().sum())






FLAGS = flags.FLAGS
flags.DEFINE_integer('nb_epochs', 20, 'Number of epochs to train model')
flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
flags.DEFINE_float('learning_rate', 0.1, 'Learning rate for training')
flags.DEFINE_integer('nb_classes', 5, 'Number of classification classes')
flags.DEFINE_integer('source_samples', 10, 'Nb of test set examples to attack')
full = pd.concat([df,dft])
assert full.shape[0] == df.shape[0] + dft.shape[0]
print("Initial test and training data shapes:", df.shape , dft.shape)

full['label'] = full['Cat']
print("Unique labels", full.label.unique())
full2 = pd.get_dummies(full , drop_first=False)
features = list(full2.columns[:-5])


y_train = np.array(full2[0:df.shape[0]][['label_Normal', 'label_Mirai', 'label_Scan', 'label_DoS', 'label_MITM ARP Spoofing']])
X_train = full2[0:df.shape[0]][features]
y_test = np.array(full2[df.shape[0]:][['label_Normal', 'label_Mirai', 'label_Scan', 'label_DoS', 'label_MITM ARP Spoofing']])
X_test = full2[df.shape[0]:][features]


scaler = MinMaxScaler().fit(X_train)
X_train_scaled = np.array(scaler.transform(X_train))
X_test_scaled = np.array(scaler.transform(X_test))
labels = full.label.unique()
le = LabelEncoder()
le.fit(labels)

y_full = le.transform(full.label)
y_train_l = y_full[0:df.shape[0]]
y_test_l = y_full[df.shape[0]:]

print("Training dataset shape", X_train_scaled.shape , y_train.shape)
print("Test dataset shape", X_test_scaled.shape , y_test.shape)             
print("Label encoder y shape", y_train_l.shape , y_test_l.shape)


def model():

	model = Sequential()
	model.add(Dense(256,activation='relu', input_shape =( X_train_scaled.shape[1],)))
	model.add(Dropout(0.4))
	model.add(Dense(256,  activation='relu'))
	model.add(Dropout(0.4))
	model.add(Dense(5 , activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics =['accuracy'])
	model.summary()
	return  model

def  evaluate():
	eval_params = {'batch_size': 128}
	accuracy = model_eval(sess , x, y, predictions , X_test_scaled , y_test , args= eval_params)
	print('Test  accuracy  on  legitimate  test  examples: ' + str(accuracy))


x = tf.placeholder(tf.float32 , shape=(None ,X_train_scaled.shape[1]))
y = tf.placeholder(tf.float32 , shape=(None ,5))




tf.set_random_seed(42)
model = model()
sess = tf.Session()
predictions = model(x)
init = tf.global_variables_initializer()
sess.run(init)




results = np.zeros((5 , source_samples), dtype='i') 
perturbations = np.zeros((5 , source_samples), dtype='f') 
grads = jacobian_graph(predictions , x, 5) 
logits = model(x)
nb_candidate = 5
overshoot = 0.02
max_iter = 10
clip_min = 0
clip_max = 1
nb_classes = len(labels)

df_params = {'nb_candidate':5, 'overshoot':0.2, 'max_iter':20, 'clip_min':0, 'clip_max':1}
print("xtest shape", X_test_scaled.shape)

#KerasClassifier to train model
classifier = KerasClassifier(model=model)
classifier.fit(X_train_scaled, y_train, nb_epochs=20, batch_size=128)

preds = np.argmax(classifier.predict(X_test_scaled), axis=1)
acc = np.sum(preds == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("\nTest accuracy: %.2f%%" % (acc * 100))


#DeepFool
adv_crafter = DeepFool(classifier, max_iter=4)
x_test_adv = adv_crafter.generate(X_test_scaled, eps=0.001)


preds = np.argmax(classifier.predict(x_test_adv), axis=1)
acc = np.sum(np.equal(preds, np.argmax(y_test, axis=1))) / y_test.shape[0]
logger.info("Classifier before adversarial training")
logger.info("Accuracy on adversarial samples: %.2f%%", (acc * 100))





#kmeans
print('kmeans')
km = KMeans(n_clusters=5)

km.fit(X_train_scaled)
y_pred = km.predict(X_test_scaled)
normalized_mutual_info_score(y_test_l, y_pred)
print('Accuracy score of kmeans:', accuracy_score(y_test_l, y_pred))

#predict using adversarial test samples
x_test_adv = x_test_adv.astype('float')
pred_adv = km.predict(x_test_adv)
print (" Accuracy score adversarial :", accuracy_score(y_test_l, pred_adv))





##Decision tree
print('Decision tree')

dt = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
dt.fit(X_train_scaled, y_train)
y_pred = dt.predict(X_test_scaled)

                 
print("Array Dimension = ",len(y_pred.shape))                
fpr_dt, tpr_dt,_ = roc_curve(y_test[:, 0], y_pred[:, 0])

roc_auc_dt = auc(fpr_dt, tpr_dt)
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average='micro'))
print('AUC score:', roc_auc_dt)


#predict using adversarial test samples
y_pred_adv = dt.predict(x_test_adv)
fpr_dt_adv, tpr_dt_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_dt_adv = auc(fpr_dt_adv, tpr_dt_adv)
print (" Accuracy score adversarial :", accuracy_score(y_test, y_pred_adv))
print ("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro'))
print (" AUC score adversarial :", roc_auc_dt_adv)

plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color ='darkorange', lw=lw, label='ROC curve (area = %0.2f)' %  roc_auc_dt)
plt.plot(fpr_dt_adv, tpr_dt_adv, color ='green',lw =lw, label ='ROC curve adv.(area = %0.2f)' % roc_auc_dt_adv)
plt.plot([0,1],[0,1], color ='navy', lw =lw, linestyle ='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Decision Tree(class = Normal)')
plt.legend(loc ="lower right")
plt.savefig('ROC_DT.png')

############################################################

print('SVM!!')

sv = OneVsRestClassifier(LinearSVC(C=1., random_state=42, loss='hinge'))
sv.fit(X_train_scaled, y_train)
y_pred = sv.predict(X_test_scaled)

                 
                 
fpr_sv, tpr_sv,_ = roc_curve(y_test[:, 0], y_pred[:, 0])

roc_auc_sv = auc(fpr_sv, tpr_sv)
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average='micro'))
print('AUC score:', roc_auc_dt)

#predict using adversarial test samples
y_pred_adv = sv.predict(x_test_adv)
fpr_sv_adv, tpr_sv_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_sv_adv = auc(fpr_sv_adv, tpr_sv_adv)
print (" Accuracy score adversarial :", accuracy_score(y_test, y_pred_adv))
print ("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro'))
print (" AUC score adversarial :", roc_auc_dt_adv)

plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color ='darkorange', lw=lw, label='ROC curve (area = %0.2f)' %  roc_auc_dt)
plt.plot(fpr_dt_adv, tpr_dt_adv, color ='green',lw =lw, label ='ROC curve adv.(area = %0.2f)' % roc_auc_dt_adv)
plt.plot([0,1],[0,1], color ='navy', lw =lw, linestyle ='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC SVM(class = Normal)')
plt.legend(loc ="lower right")
plt.savefig('ROC_SVM.png')

################################################################


print('k-nearest!!')

#dt = OneVsRestClassifier(DecisionTreeClassifier(random_state=42))
#dt.fit(X_train_scaled, y_train)
from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=1, n_jobs=-1)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)

                 
                 
fpr_dt, tpr_dt,_ = roc_curve(y_test[:, 0], y_pred[:, 0])

roc_auc_dt = auc(fpr_dt, tpr_dt)
print('Accuracy score:', accuracy_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred, average='micro'))
print('AUC score:', roc_auc_dt)



#predict using adversarial test samples
y_pred_adv = classifier.predict(x_test_adv)
fpr_dt_adv, tpr_dt_adv, _ = roc_curve(y_test[:, 0], y_pred_adv[:, 0])
roc_auc_dt_adv = auc(fpr_dt_adv, tpr_dt_adv)
print (" Accuracy score adversarial :", accuracy_score(y_test, y_pred_adv))
print ("F1 score adversarial :", f1_score(y_test, y_pred_adv, average='micro'))
print (" AUC score adversarial :", roc_auc_dt_adv)

plt.figure()
lw = 2
plt.plot(fpr_dt, tpr_dt, color ='darkorange', lw=lw, label='ROC curve (area = %0.2f)' %  roc_auc_dt)
plt.plot(fpr_dt_adv, tpr_dt_adv, color ='green',lw =lw, label ='ROC curve adv.(area = %0.2f)' % roc_auc_dt_adv)
plt.plot([0,1],[0,1], color ='navy', lw =lw, linestyle ='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC K-Nearest(class = Normal)')
plt.legend(loc ="lower right")
plt.savefig('ROC_KN.png')






