# 10-fold CV of SVM on the dataset with 4 classes
# 'HVHA', 'HVLA', 'LVHA', 'LVLA'
from __future__ import print_function

import mat73
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 16})

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# Importing the data files
class1 = mat73.loadmat('../DATA/PSD_ALL1.mat')
class2 = mat73.loadmat('../DATA/PSD_ALL2.mat')
class3 = mat73.loadmat('../DATA/PSD_ALL3.mat')
class4 = mat73.loadmat('../DATA/PSD_ALL4.mat')
# Taking absolute value of the data (since it is complex)
class1 = abs(class1['PSD_ALL_1'])
class2 = abs(class2['PSD_ALL_2'])
class3 = abs(class3['PSD_ALL_3'])
class4 = abs(class4['PSD_ALL_4'])

print(class1.shape)
print(class2.shape)
print(class3.shape)
print(class4.shape)

# Combine data from all classes and reshape such that trials are in the first dimension
# Take the mean over the time dimension
X1 = np.concatenate((class1, class2, class3, class4), axis=3)
X1 = np.transpose(X1, [3, 0, 1, 2])
X1.shape
X1 = np.mean(X1, axis=2)

# Extracting beta, lower gamma and higher gamma bands and averaging each separately
temp1 = np.mean(X1[:, 13:30, :], axis=1)
temp2 = np.mean(X1[:, 31:50, :], axis=1)
temp3 = np.mean(X1[:, 51:79, :], axis=1)

X = np.concatenate((temp1, temp2, temp2), axis=1)
print(X.shape)

# Forming the label vector
y1 = 0 * np.ones((class1.shape[3]))
y2 = 1 * np.ones((class2.shape[3]))
y3 = 2 * np.ones((class3.shape[3]))
y4 = 3 * np.ones((class4.shape[3]))

y = np.concatenate((y1, y2, y3, y4), axis=0).astype(int)

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

from sklearn.svm import SVC
from sklearn.decomposition import PCA

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
plt.savefig('FIGURES/pca_param_select_svm_4cl.pdf')

print('Optimal num PCs = %d' % (npc_opt))
print('Optimal gamma   = %f' % (gam_opt))

npc = npc_opt
gam = gam_opt

## 10 fold cross validation with optimal parameters npcs = 25 gamma = 0.1
# Create cross-validation object
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

classNames = ['HVLA', 'HVHA', 'LVLA', 'LVHA']
n_classes = 4
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
accSVM_4_fn = ('REPORTS/accuracySVM_4.p')
with open(accSVM_4_fn, 'wb') as fp:
    pickle.dump(acc_mean, fp)

reports_4_fn = ('REPORTS/reportsSVM_4.p')
with open(reports_4_fn, 'wb') as fp:
    pickle.dump(reports, fp)

confusionMatrices_4_fn = ('REPORTS/confusionMatricesSVM_4.p')
with open(confusionMatrices_4_fn, 'wb') as fp:
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
    plt.savefig('FIGURES/accuracy_svm_10fold_4_classes.pdf')

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
ax.set_yticklabels(('HVLA', 'HVHA', 'LVLA', 'LVHA'), rotation=90, fontsize="16", va="center")
plt.tight_layout()
plt.savefig('FIGURES/normCM_svm_10fold_4_classes.pdf')
