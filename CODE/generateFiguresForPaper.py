#  for plotting some figures
from __future__ import print_function

import pickle

import mat73
import matplotlib
import numpy as np

import matplotlib.pyplot as plt
from pylab import plot, show, savefig, xlim, figure, ylim, legend, boxplot, setp, axes
######################################################################################################################################
# read the recordings

reportsCNN_4_fn = ('REPORTS/reportsCNN_4.p')
with open(reportsCNN_4_fn, 'rb') as fp:
    reportsCNN_4 = pickle.load(fp)

reportsCNN_12_fn = ('REPORTS/reportsCNN_12.p')
with open(reportsCNN_12_fn, 'rb') as fp:
    reportsCNN_12 = pickle.load(fp)

reportsSVM_4_fn = ('REPORTS/reportsSVM_4.p')
with open(reportsSVM_4_fn, 'rb') as fp:
    reportsSVM_4 = pickle.load(fp)

reportsSVM_12_fn = ('REPORTS/reportsSVM_12.p')
with open(reportsSVM_12_fn, 'rb') as fp:
    reportsSVM_12 = pickle.load(fp)

# 4 classes
reportsDictSVM4 = reportsSVM_4
reportsDictCNN4 = reportsCNN_4
numFolds = 10
numClasses = 4
avgPrecisionSVM4 = np.zeros(numClasses)
avgRecallSVM4 = np.zeros(numClasses)
avgF1SVM4 = np.zeros(numClasses)

avgPrecisionCNN4 = np.zeros(numClasses)
avgRecallCNN4 = np.zeros(numClasses)
avgF1CNN4 = np.zeros(numClasses)

classes = list(reportsDictSVM4[0].keys())[0:numClasses]
for clIdx in range(len(classes)):
    tmpPrecisionSVM4 = []
    tmpRecallSVM4 = []
    tmpF1SVM4 = []

    tmpPrecisionCNN4 = []
    tmpRecallCNN4 = []
    tmpF1CNN4 = []

    for f in range(numFolds):
        tmpPrecisionSVM4.append(reportsDictSVM4[f][classes[clIdx]]['precision'])
        tmpRecallSVM4.append(reportsDictSVM4[f][classes[clIdx]]['recall'])
        tmpF1SVM4.append(reportsDictSVM4[f][classes[clIdx]]['f1-score'])

        tmpPrecisionCNN4.append(reportsDictCNN4[f][classes[clIdx]]['precision'])
        tmpRecallCNN4.append(reportsDictCNN4[f][classes[clIdx]]['recall'])
        tmpF1CNN4.append(reportsDictCNN4[f][classes[clIdx]]['f1-score'])

    avgPrecisionSVM4[clIdx] = np.mean(tmpPrecisionSVM4)
    avgRecallSVM4[clIdx] = np.mean(tmpRecallSVM4)
    avgF1SVM4[clIdx] = np.mean(tmpF1SVM4)

    avgPrecisionCNN4[clIdx] = np.mean(tmpPrecisionCNN4)
    avgRecallCNN4[clIdx] = np.mean(tmpRecallCNN4)
    avgF1CNN4[clIdx] = np.mean(tmpF1CNN4)

# 12 classes
reportsDictSVM12 = reportsSVM_12
reportsDictCNN12 = reportsCNN_12
numFolds = 10
numClasses = 12
avgPrecisionSVM12 = np.zeros(numClasses)
avgRecallSVM12 = np.zeros(numClasses)
avgF1SVM12 = np.zeros(numClasses)

avgPrecisionCNN12 = np.zeros(numClasses)
avgRecallCNN12 = np.zeros(numClasses)
avgF1CNN12 = np.zeros(numClasses)

classes = list(reportsDictSVM12[0].keys())[0:numClasses]
for clIdx in range(len(classes)):
    tmpPrecisionSVM12 = []
    tmpRecallSVM12 = []
    tmpF1SVM12 = []

    tmpPrecisionCNN12 = []
    tmpRecallCNN12 = []
    tmpF1CNN12 = []

    for f in range(numFolds):
        tmpPrecisionSVM12.append(reportsDictSVM12[f][classes[clIdx]]['precision'])
        tmpRecallSVM12.append(reportsDictSVM12[f][classes[clIdx]]['recall'])
        tmpF1SVM12.append(reportsDictSVM12[f][classes[clIdx]]['f1-score'])

        tmpPrecisionCNN12.append(reportsDictCNN12[f][classes[clIdx]]['precision'])
        tmpRecallCNN12.append(reportsDictCNN12[f][classes[clIdx]]['recall'])
        tmpF1CNN12.append(reportsDictCNN12[f][classes[clIdx]]['f1-score'])

    avgPrecisionSVM12[clIdx] = np.mean(tmpPrecisionSVM12)
    avgRecallSVM12[clIdx] = np.mean(tmpRecallSVM12)
    avgF1SVM12[clIdx] = np.mean(tmpF1SVM12)

    avgPrecisionCNN12[clIdx] = np.mean(tmpPrecisionCNN12)
    avgRecallCNN12[clIdx] = np.mean(tmpRecallCNN12)
    avgF1CNN12[clIdx] = np.mean(tmpF1CNN12)


svm4_means, svm4_std = (np.mean(avgPrecisionSVM4), np.mean(avgRecallSVM4),np.mean(avgF1SVM4)),(np.std(avgPrecisionSVM4),np.std(avgRecallSVM4),np.std(avgF1SVM4))
svm12_means, svm12_std = (np.mean(avgPrecisionSVM12), np.mean(avgRecallSVM12),np.mean(avgF1SVM12)), (np.std(avgPrecisionSVM12), np.std(avgRecallSVM12),np.std(avgF1SVM12))


#########################################################################################################################
## plot avg precision recall F1 barplot
# function for setting the colors of the box plots pairs
plt.style.use('default')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def setBoxColors(bp):
    setp(bp['boxes'][0],facecolor='blue',alpha=0.5)
    setp(bp['boxes'][0], color='blue')
    setp(bp['caps'][0], color='blue')
    setp(bp['caps'][1], color='blue')
    setp(bp['whiskers'][0], color='blue')
    setp(bp['whiskers'][1], color='blue')
    setp(bp['fliers'][0], color='blue')
    setp(bp['fliers'][1], color='blue')
    setp(bp['medians'][0], color='blue')

    setp(bp['boxes'][1], facecolor='darkgreen', alpha=0.6)
    setp(bp['boxes'][1], color='darkgreen')
    setp(bp['caps'][2], color='darkgreen')
    setp(bp['caps'][3], color='darkgreen')
    setp(bp['whiskers'][2], color='darkgreen')
    setp(bp['whiskers'][3], color='darkgreen')
    setp(bp['fliers'][2], color='darkgreen')
    setp(bp['fliers'][3], color='darkgreen')
    setp(bp['medians'][1], color='darkgreen')

    setp(bp['boxes'][2], facecolor='red', alpha=0.6)
    setp(bp['boxes'][2], color='red')
    setp(bp['caps'][4], color='red')
    setp(bp['caps'][5], color='red')
    setp(bp['whiskers'][4], color='red')
    setp(bp['whiskers'][5], color='red')
    setp(bp['fliers'][2], color='red')
    setp(bp['fliers'][3], color='red')
    setp(bp['medians'][2], color='red')

    setp(bp['boxes'][3], facecolor='darkorange', alpha=0.6)
    setp(bp['boxes'][3], color='darkorange')
    setp(bp['caps'][6], color='darkorange')
    setp(bp['caps'][7], color='darkorange')
    setp(bp['whiskers'][6], color='darkorange')
    setp(bp['whiskers'][7], color='darkorange')
    setp(bp['fliers'][3], color='darkorange')
    setp(bp['fliers'][3], color='darkorange')
    setp(bp['medians'][3], color='darkorange')

A= [avgPrecisionSVM4,  avgPrecisionSVM12 , avgPrecisionCNN4,avgPrecisionCNN12 ]
B = [avgRecallSVM4,  avgRecallSVM12, avgRecallCNN4,avgRecallCNN12 ]
C = [avgF1SVM4, avgF1SVM12 , avgF1CNN4,avgF1CNN12]

fig = figure(figsize=(8, 5))#height was 5.5
ax = axes()

# first boxplot pair
bp = boxplot(A, positions = [1.3, 2.1, 2.9, 3.7], widths = 0.8,patch_artist=True, sym='+')
setBoxColors(bp)

# second boxplot pair
bp = boxplot(B, positions = [7.3, 8.1, 8.9, 9.7], widths = 0.8,patch_artist=True, sym='+')
setBoxColors(bp)

# thrid boxplot pair
bp = boxplot(C, positions = [13.3, 14.1,14.9,15.7], widths = 0.8,patch_artist=True, sym='+')
setBoxColors(bp)

# set axes limits and labels
xlim(0,17)
ylim(0.1,1)
ax.set_xticklabels(['Precision', 'Recall', 'F1-score'])
ax.set_xticks([2.5, 8.5, 14.5])

# draw temporary red and blue lines and use them to create a legend
hSVM4, = plot([1,1],'b-')
hSVM12, = plot([1,1],color= 'darkgreen')
hCNN4, = plot([1,1],color= 'red')
hCNN12, = plot([1,1],color= 'darkorange')
legend((hSVM4, hSVM12,hCNN4,hCNN12),('SVM 4 classes', 'SVM 12 classes','CNN 4 classes', 'CNN 12 classes'))
hSVM4.set_visible(False)
hSVM12.set_visible(False)
hCNN4.set_visible(False)
hCNN12.set_visible(False)
fig.tight_layout()
plt.savefig('FIGURES/metrics_10fold_barplot_paper.pdf')

########################################################################################################################
## plot confusion matrices
import pandas as pd
import seaborn.apionly as sns
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
##12 classes SVM
classNames = ['PVHA_i1', 'PVHA_i2', 'PVHA_i3',
              'PVLA_i1', 'PVLA_i2', 'PVLA_i3',
              'NVHA_i1', 'NVHA_i2', 'NVHA_i3',
              'NVLA_i1', 'NVLA_i2', 'NVLA_i3',
              ]
n_classes = 12
nfold = 10
confusionMatrices_12_fn = 'REPORTS/confusionMatricesSVM_12.p'
with open(confusionMatrices_12_fn, 'rb') as fp:
    confusionMatrices = pickle.load(fp)

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(9.6, 4.7 ))#5.7
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 14}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.tick_params(axis='x', which='major', pad=-3)
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="13", va="center")
ax.set_xticklabels(classNames,rotation = 45,fontsize="13", ha="right")
plt.tight_layout()
plt.savefig('FIGURES/normCM_svm_10fold_12_classes_new.pdf')


##12 classes CNN
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
classNames = ['PVHA_i1', 'PVHA_i2', 'PVHA_i3',
              'PVLA_i1', 'PVLA_i2', 'PVLA_i3',
              'NVHA_i1', 'NVHA_i2', 'NVHA_i3',
              'NVLA_i1', 'NVLA_i2', 'NVLA_i3',
              ]
n_classes = 12
nfold = 10
confusionMatrices_12_fn = 'REPORTS/confusionMatricesCNN_12.p'
with open(confusionMatrices_12_fn, 'rb') as fp:
    confusionMatrices = pickle.load(fp)

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(9.6, 4.7))
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 14}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.tick_params(axis='x', which='major', pad=-3)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="13", va="center")
ax.set_xticklabels(classNames,rotation = 45,fontsize="13", ha="right")
plt.tight_layout()
plt.savefig('FIGURES/normCM_cnn_10fold_12_classes_new.pdf')


# SVM 4 classes
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
classNames = ['PVLA', 'PVHA', 'NVLA', 'NVHA']
n_classes = 4
confusionMatrices_4_fn = ('REPORTS/confusionMatricesSVM_4.p')
with open(confusionMatrices_4_fn, 'rb') as fp:
    confusionMatrices = pickle.load(fp)

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(9.6, 4.1))# height 5.7
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 18}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="16", va="center")
ax.set_xticklabels(classNames, rotation=0, fontsize="16", ha="center")
plt.tight_layout()
plt.savefig('FIGURES/normCM_svm_10fold_4_classes_new.pdf')

# CNN 4 classes
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
classNames = ['PVLA', 'PVHA', 'NVLA', 'NVHA']
n_classes = 4
confusionMatrices_4_fn = ('REPORTS/confusionMatricesCNN_4.p')
with open(confusionMatrices_4_fn, 'rb') as fp:
    confusionMatrices = pickle.load(fp)

# normalize confusion matrices
normalizedAvgCM = np.zeros((n_classes, n_classes))
for i in range(len(confusionMatrices)):
    cm = confusionMatrices[i]
    normalizedAvgCM += cm / cm.astype(np.float).sum(axis=1)

normalizedAvgCM = normalizedAvgCM / nfold
# plot one time prediction confusion matrix
df_cm = pd.DataFrame(normalizedAvgCM, index=classNames, columns=classNames)
plt.figure(figsize=(9.6,4.1 ))#5.7
sns.set(font_scale=1.4)  # for label size
ax = sns.heatmap(df_cm, cbar_kws={'ticks': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]}, vmin=0, vmax=1.0,
                 annot=True, annot_kws={"size": 18}, fmt='2.2f', cmap="Blues")  # font size
bottom, top = ax.get_ylim()
ax.set_ylim(bottom + 0.5, top - 0.5)
ax.set_ylim(sorted(ax.get_xlim(), reverse=True))
ax.set_yticklabels(classNames, rotation=0, fontsize="16", va="center")
ax.set_xticklabels(classNames, rotation=0, fontsize="16", ha="center")
plt.tight_layout()
plt.savefig('FIGURES/normCM_cnn_10fold_4_classes_new.pdf')
