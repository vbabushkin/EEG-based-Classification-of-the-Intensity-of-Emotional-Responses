# 10-fold CV of SVM on the dataset with 12 classes
# PVHA_0, PVLA_0, LVHA_0, NVLA_0, 'PVHA_1', 'PVHA_2', 'PVLA_1', 'PVLA_2', 'LVHA_1', 'LVHA_2','NVLA_1', 'NVLA_2'
from __future__ import print_function

import mat73
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 16})
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# data format:

# 80 freq. bins X 160 time X 59 channels X trials
#
# Trials: 34 participants X 20 pictures = 680 trials
# 182   + 225  +  273 = 680
# 181   + 236  +  263 = 680
# 295   + 206  +  179 = 680
# 48    +  76  +  556 = 680


#Importing the data files
class1 = mat73.loadmat('../DATA_RATING/PSD_HVHA_0.mat')
class2 = mat73.loadmat('../DATA_RATING/PSD_HVHA_1.mat')
class3 = mat73.loadmat('../DATA_RATING/PSD_HVHA_2.mat')
class4 = mat73.loadmat('../DATA_RATING/PSD_HVLA_0.mat')
class5 = mat73.loadmat('../DATA_RATING/PSD_HVLA_1.mat')
class6 = mat73.loadmat('../DATA_RATING/PSD_HVLA_2.mat')
class7 = mat73.loadmat('../DATA_RATING/PSD_LVHA_0.mat')
class8 = mat73.loadmat('../DATA_RATING/PSD_LVHA_1.mat')
class9 = mat73.loadmat('../DATA_RATING/PSD_LVHA_2.mat')
class10 = mat73.loadmat('../DATA_RATING/PSD_LVLA_0.mat')
class11 = mat73.loadmat('../DATA_RATING/PSD_LVLA_1.mat')
class12 = mat73.loadmat('../DATA_RATING/PSD_LVLA_2.mat')
#Taking absolute value of the data (since it is complex)
class1=abs(class1['PSD_ALL_4'])
class2=abs(class2['PSD_ALL_5'])
class3=abs(class3['PSD_ALL_6'])
class4=abs(class4['PSD_ALL_1'])
class5=abs(class5['PSD_ALL_2'])
class6=abs(class6['PSD_ALL_3'])
class7=abs(class7['PSD_ALL_10'])
class8=abs(class8['PSD_ALL_11'])
class9=abs(class9['PSD_ALL_12'])
class10=abs(class10['PSD_ALL_7'])
class11=abs(class11['PSD_ALL_8'])
class12=abs(class12['PSD_ALL_9'])
print(class1.shape)
print(class2.shape)
print(class3.shape)
print(class4.shape)
print(class5.shape)
print(class6.shape)
print(class7.shape)
print(class8.shape)
print(class9.shape)
print(class10.shape)
print(class11.shape)
print(class12.shape)

#Combine data from all classes and reshape such that trials are in the first dimension
#Take the mean over the time dimension
X1=np.concatenate((class1,class2,class3,class4,class5,class6,class7,class8,class9,class10,class11,class12),axis=3)
X1=np.transpose(X1, [3,0,1,2])
X1.shape
X1=np.mean(X1,axis=2)

#Extracting beta, lower gamma and higher gamma bands and averaging each separately
temp1=np.mean(X1[:,13:30,:],axis=1)
temp2=np.mean(X1[:,31:50,:],axis=1)
temp3=np.mean(X1[:,51:79,:],axis=1)

X=np.concatenate((temp1,temp2,temp2),axis=1)
print(X.shape)


#Forming the label vector
y1=0*np.ones((class1.shape[3]))
y2=1*np.ones((class2.shape[3]))
y3=2*np.ones((class3.shape[3]))
y4=3*np.ones((class4.shape[3]))
y5=4*np.ones((class5.shape[3]))
y6=5*np.ones((class6.shape[3]))
y7=6*np.ones((class7.shape[3]))
y8=7*np.ones((class8.shape[3]))
y9=8*np.ones((class9.shape[3]))
y10=9*np.ones((class10.shape[3]))
y11=10*np.ones((class11.shape[3]))
y12=11*np.ones((class12.shape[3]))

y=np.concatenate((y1,y2,y3,y4,y5,y6,y7,y8,y9,y10,y11,y12),axis=0).astype(int)

# split on testing and training datasets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.20, random_state=2)

# determine optimal parameters:  gamma and number of components
npc_test = [5, 10, 25, 50, 75, 100, 125, 150]
gam_test = [1e-4, 4e-4, 1e-3, 4e-3, 1e-2, 1e-1, 1, 10]
C = 100
n0 = len(npc_test)
n1 = len(gam_test)
acc = np.zeros((n0, n1))
acc_max = 0

for i0, npc in enumerate(npc_test):

    # Fit PCA on the training data
    pca = PCA(n_components=npc, svd_solver='randomized', whiten=True)
    pca.fit(Xtr)

    # Transform the training and test
    Ztr = pca.transform(Xtr)
    Zts = pca.transform(Xts)

    for i1, gam in enumerate(gam_test):

        # Fiting on the transformed training data
        svc = SVC(C=C, kernel='rbf', gamma=gam)
        svc.fit(Ztr, ytr)

        # Predict on the test data
        yhat = svc.predict(Zts)

        # Compute the accuracy
        acc[i0, i1] = np.mean(yhat == yts)
        print('npc=%d gam=%12.4e acc=%12.4e' % (npc, gam, acc[i0, i1]))

        # Save the optimal parameters
        if acc[i0, i1] > acc_max:
            gam_opt = gam
            npc_opt = npc
            acc_max = acc[i0, i1]

df_acc = pd.DataFrame(acc, index=npc_test, columns=gam_test)
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_acc, cbar_kws={'ticks': [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}, vmin=0.2, vmax=0.9, annot=True,
                 annot_kws={"size": 12}, fmt='f', cmap="Reds")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom, top)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(npc_test,
                   rotation=90, fontsize="16", va="center")
plt.imshow(acc, aspect='auto', cmap='Reds')
plt.xlabel('Gamma')
plt.ylabel('Num PCs')
plt.tight_layout()
plt.savefig('FIGURES/pca_param_select_svm_12cl.pdf')

print('Optimal num PCs = %d' % (npc_opt))
print('Optimal gamma   = %f' % (gam_opt))

npc = npc_opt
gam = gam_opt

## 10 fold cross validation with optimal parameters npcs = 25 gamma = 0.1
# Create cross-validation object
classNames = ['PVHA_0', 'PVHA_1', 'PVHA_2',
              'PVLA_0', 'PVLA_1', 'PVLA_2',
              'LVHA_0', 'LVHA_1', 'LVHA_2',
              'NVLA_0', 'NVLA_1', 'NVLA_2',
              ]
n_classes = 12
nfold = 10
kf = KFold(nfold, shuffle=True)

# Create the scaler objects
xscal = StandardScaler()
yscal = StandardScaler()

# Run the cross-validation
rsq = np.zeros(nfold)
acc = np.zeros(nfold)
yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatrices = []
reports = []
for ifold, ind in enumerate(kf.split(X)):
    print('Fold = %d' % ifold)
    # Get the training data in the split
    Itr, Its = ind
    Xtr = X[Itr, :]
    ytr = y[Itr]
    Xts = X[Its, :]
    yts = y[Its]

    # Fit and transform the data
    Xtr1 = xscal.fit_transform(Xtr)
    Xts1 = xscal.transform(Xts)

    # Fit PCA on the training data
    pca = PCA(n_components=25, svd_solver='randomized', whiten=True)
    pca.fit(Xtr1)

    # Transform the training and test
    Ztr = pca.transform(Xtr1)
    Zts = pca.transform(Xts1)

    # Fiting on the transformed training data
    svc = SVC(C=C, kernel='rbf', gamma=0.1)
    svc.fit(Ztr, ytr)

    # Predict on the test data
    yhat = svc.predict(Zts)
    yhatPerFold.append(yhat)
    rsq[ifold] = r2_score(yts, yhat)
    acc[ifold] = accuracy_score(yts, yhat)

    print('R^2     = %12.4e' % rsq[ifold])
    print('acc     = %12.4e' % acc[ifold])

    a = classification_report(yts, yhat, target_names=classNames, output_dict=True)
    reports.append(a)
    print("Confusion matrix on the test data")
    cm = confusion_matrix(yts, yhat, labels=range(n_classes))
    print(cm)
    confusionMatrices.append(cm)

acc_mean = []
# Compute mean accuracy
for i in range(nfold):
    acc_mean.append(reports[i]['accuracy'])

import pickle

# Save history
accSVM_12_fn = ('REPORTS/accuracySVM_12.p')
with open(accSVM_12_fn, 'wb') as fp:
    pickle.dump(acc_mean, fp)

reports_12_fn = ('REPORTS/reportsSVM_12.p')
with open(reports_12_fn, 'wb') as fp:
    pickle.dump(reports, fp)

confusionMatrices_12_fn = ('REPORTS/confusionMatricesSVM_12.p')
with open(confusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(confusionMatrices, fp)

with plt.style.context("default"):
    plt.figure(figsize=(8, 6))
    plt.plot(np.linspace(1, 10, 10), acc_mean, label='accuracy')
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.title("Accuracy for each fold", fontsize=18)
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid()
    plt.savefig('FIGURES/accuracy_svm_10fold_12_classes.pdf')

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(10, 7))
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 12}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="16", va="center")
plt.tight_layout()
plt.savefig('FIGURES/normCM_svm_10fold_12_classes.pdf')
