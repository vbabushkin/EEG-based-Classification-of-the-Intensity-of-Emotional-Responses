# 10-fold CV of CNN on the dataset with 12 classes
# HVHA_0, HVLA_0, LVHA_0, LVLA_0, 'HVHA_1', 'HVHA_2', 'HVLA_1', 'HVLA_2', 'LVHA_1', 'LVHA_2','LVLA_1', 'LVLA_2'
from __future__ import print_function

import mat73
import matplotlib
import numpy as np

matplotlib.rcParams.update({'font.size': 16})

import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import optimizers
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns

# Importing the data files
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
# Taking absolute value of the data (since it is complex)
class1 = abs(class1['PSD_ALL_4'])
class2 = abs(class2['PSD_ALL_5'])
class3 = abs(class3['PSD_ALL_6'])
class4 = abs(class4['PSD_ALL_1'])
class5 = abs(class5['PSD_ALL_2'])
class6 = abs(class6['PSD_ALL_3'])
class7 = abs(class7['PSD_ALL_10'])
class8 = abs(class8['PSD_ALL_11'])
class9 = abs(class9['PSD_ALL_12'])
class10 = abs(class10['PSD_ALL_7'])
class11 = abs(class11['PSD_ALL_8'])
class12 = abs(class12['PSD_ALL_9'])
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

# Combine data from all classes and reshape such that trials are in the first dimension
# Take the mean over the time dimension
X = np.concatenate((class1, class2, class3, class4, class5, class6, class7, class8, class9, class10, class11, class12),
                   axis=3)
X = np.transpose(X, [3, 0, 1, 2])
X.shape
X = np.mean(X, axis=2)

# we need to add additional dimension to X to make it resembling pictures dataset
X = np.expand_dims(X, 3)
print(X.shape)

# Forming the label vector
y1 = 0 * np.ones((class1.shape[3]))
y2 = 1 * np.ones((class2.shape[3]))
y3 = 2 * np.ones((class3.shape[3]))
y4 = 3 * np.ones((class4.shape[3]))
y5 = 4 * np.ones((class5.shape[3]))
y6 = 5 * np.ones((class6.shape[3]))
y7 = 6 * np.ones((class7.shape[3]))
y8 = 7 * np.ones((class8.shape[3]))
y9 = 8 * np.ones((class9.shape[3]))
y10 = 9 * np.ones((class10.shape[3]))
y11 = 10 * np.ones((class11.shape[3]))
y12 = 11 * np.ones((class12.shape[3]))

y = np.concatenate((y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11, y12), axis=0).astype(int)

# split on testing and training datasets
Xtr, Xts, ytr, yts = train_test_split(X, y, test_size=0.20, random_state=2)  # 42

print('Xtr shape:  ' + str(Xtr.shape))
print('Xts shape:  ' + str(Xts.shape))

Xtr = Xtr.astype('float32')
Xts = Xts.astype('float32')



def create_mod(use_dropout=False, use_bn=False):
    num_classes = 12
    model = Sequential()
    model.add(Conv2D(32, (3, 3),  # 64,(9,9)
                     padding='valid', activation='relu',
                     input_shape=Xtr.shape[1:]))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    if use_bn:
        model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    if use_bn:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu'))
    if use_bn:
        model.add(BatchNormalization())
    if use_dropout:
        model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def create_datagen():
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # do not rotate
        horizontal_flip=False,  # do not flip
        vertical_flip=False,  # do not flip
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)
    return datagen


# Parameters
nepochs = 100
batch_size = 32
lr = 1e-3
decay = 1e-4

## 10 fold cross validation with optimal parameters npcs = 25 gamma = 0.1
# Create cross-validation object
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, accuracy_score

classNames = ['HVHA_0', 'HVHA_1', 'HVHA_2',
              'HVLA_0', 'HVLA_1', 'HVLA_2',
              'LVHA_0', 'LVHA_1', 'LVHA_2',
              'LVLA_0', 'LVLA_1', 'LVLA_2',
              ]
n_classes = 12
nfold = 10

# plot train/test distribution
with plt.style.context("default"):
    fig, axs = plt.subplots(2, 1, figsize=(7, 5), sharex=False, sharey=False)
    n, bins, patches = axs[0].hist(ytr, n_classes, rwidth=0.9, alpha=0.8)
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    axs[0].set_xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w))
    axs[0].set_xlim(bins[0], bins[-1])
    axs[0].set_xticklabels(classNames, fontsize=8, rotation=0)
    axs[0].set_ylabel('Number of train instances')

    n, bins, patches = axs[1].hist(yts, n_classes, rwidth=0.9, alpha=0.8)
    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
    axs[1].set_xticks(np.arange(min(bins) + bin_w / 2, max(bins), bin_w))
    axs[1].set_xlim(bins[0], bins[-1])
    axs[1].set_xticklabels(classNames, fontsize=8, rotation=0)
    axs[1].set_ylabel('Number of test instances')
    plt.tight_layout()
    plt.savefig('FIGURES/trainTestDist_cnn_10fold_12_classes.pdf')

kf = KFold(nfold, shuffle=True)

# Run the cross-validation
rsq = np.zeros(nfold)
acc = np.zeros(nfold)
yhatPerFold = []
ytsPerFold = []
accuracyTrain = []
accuracyVal = []
confusionMatrices = []
reports = []

from sklearn.preprocessing import StandardScaler

xscal = StandardScaler()

for ifold, ind in enumerate(kf.split(X)):
    print('Fold = %d' % ifold)
    # Get the training data in the split
    Itr, Its = ind
    Xtr = X[Itr, :]
    ytr = y[Itr]
    Xts = X[Its, :]
    yts = y[Its]

    # clear session for each fold, otherwise it will be utilizing model trained from previous fold
    # and it will result in the validation accuracy values close to 1
    K.clear_session()
    model = create_mod()
    # Create the optimizer
    opt = optimizers.RMSprop(lr=lr, decay=decay)

    # Compile
    hist = model.compile(loss='sparse_categorical_crossentropy',
                         optimizer=opt,
                         metrics=['accuracy'])
    print(model.summary())

    # Fit and transform the data
    Xtr = xscal.fit_transform(Xtr.reshape(-1, Xtr.shape[-1])).reshape(Xtr.shape)
    Xts = xscal.fit_transform(Xts.reshape(-1, Xts.shape[-1])).reshape(Xts.shape)

    # Fit the model with no data augmentation
    hist = model.fit(Xtr, ytr, batch_size=batch_size,
                     epochs=nepochs, validation_data=(Xts, yts),
                     shuffle=True)
    hist_dict = hist.history
    testAcc = hist_dict['acc']
    valAcc = hist_dict['val_acc']
    accuracyTrain.append(testAcc)
    accuracyVal.append(valAcc)

    yhat = model.predict(Xts)
    labBin = LabelBinarizer()
    labBin.fit(yts)
    yhat1 = labBin.inverse_transform(np.round(yhat))

    yhat = yhat1

    yhatPerFold.append(yhat)
    ytsPerFold.append(yts)
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
reports_12_fn = ('REPORTS/reportsCNN_12.p')
with open(reports_12_fn, 'wb') as fp:
    pickle.dump(reports, fp)

confusionMatrices_12_fn = ('REPORTS/confusionMatricesCNN_12.p')
with open(confusionMatrices_12_fn, 'wb') as fp:
    pickle.dump(confusionMatrices, fp)

# Compute mean accuracy
acc_mean_train = np.mean(accuracyTrain, axis=1)
acc_mean_val = np.mean(accuracyVal, axis=1)

with plt.style.context("default"):
    plt.figure(figsize=(10, 8))
    plt.plot(np.linspace(1, 10, 10), acc_mean_train, label='training accuracy')
    plt.plot(np.linspace(1, 10, 10), acc_mean_val, label='validation accuracy')
    plt.xlabel('folds')
    plt.ylabel('accuracy')
    plt.title("Accuracy for each fold")
    plt.legend()
    plt.xlim([0.5, 10.5])
    plt.ylim([0.4, 1.0])
    plt.xticks(np.arange(1, 10.5, step=1))
    plt.grid
    plt.savefig('FIGURES/accTrainVal_10foldCV_12classes.pdf')

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
plt.savefig('FIGURES/normCM_cnn_10fold_12_classes.pdf')

with plt.style.context("default"):
    plt.figure(figsize=(10, 5))
    for iplt in range(2):

        plt.subplot(1, 2, iplt + 1)

        if iplt == 0:
            acc = np.mean(accuracyTrain, axis=0)
        else:
            acc = np.mean(accuracyVal, axis=0)
        plt.plot(acc, '-', linewidth=3)

        n = len(acc)
        nepochs = len(acc)
        plt.grid()
        plt.xlim([0, nepochs])

        plt.xlabel('Epoch')
        if iplt == 0:
            plt.ylabel('Train accuracy')
        else:
            plt.ylabel('Test accuracy')

    plt.tight_layout()
    plt.savefig('FIGURES/accuracyTrainVal_cnn_10fold_12_classes.pdf')
