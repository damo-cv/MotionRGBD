'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

# -------------------
# import modules
# -------------------
import random, os
import numpy as np
import cv2
import heapq
import shutil
from textwrap import wrap

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimage

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score
import seaborn as sns
from torchvision import transforms
from PIL import Image
import torch
from torchvision.utils import save_image, make_grid

# ---------------------------------------
# Plot Confusion Matrix
# ---------------------------------------
def plot_confusion_matrix(PREDICTIONS_PATH, grounds, preds, categories, idx, top=20):
    print("--------------------------------------------")
    print("Confusion Matrix")
    print("--------------------------------------------")

    super_category = str(idx)
    num_cat = []
    for ind, cat in enumerate(categories):
        print("Class {0} : {1}".format(ind, cat))
        num_cat.append(ind)
    print()
    numclass = len(num_cat)

    cm = confusion_matrix(grounds, preds, labels=num_cat)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot()
    sns.heatmap(cm, annot=False, fmt='g', ax=ax);  # annot=True to annotate cells, ftm='g' to disable scientific notation
    
    # labels, title and ticks
    ax.set_title('Confusion Matrix - ' + super_category, fontsize=20)
    ax.set_xlabel('Predicted labels', fontsize=16)
    ax.set_ylabel('True labels', fontsize=16)

    ax.set_xticks(range(0,len(num_cat), 1))
    ax.set_yticks(range(0,len(num_cat), 1))
    ax.xaxis.set_ticklabels(num_cat)
    ax.yaxis.set_ticklabels(num_cat)

    plt.pause(0.1)
    fig.savefig(os.path.join(PREDICTIONS_PATH, "confusion_matrix"), dpi=fig.dpi)

    # -------------------------------------------------
    # Plot Accuracy and Precision
    # -------------------------------------------------
    Accuracy = [(cm[i, i] / sum(cm[i, :])) * 100 if sum(cm[i, :]) != 0 else 0.000001 for i in range(cm.shape[0])]
    Precision = [(cm[i, i] / sum(cm[:, i])) * 100 if sum(cm[:, i]) != 0 else 0.000001 for i in range(cm.shape[1])]

    fig = plt.figure(figsize=(int((numclass*3)%300), 8))
    ax = fig.add_subplot()

    bar_width = 0.4
    x = np.arange(len(Accuracy))
    b1 = ax.bar(x, Accuracy, width=bar_width, label='Accuracy', color=sns.xkcd_rgb["pale red"], tick_label=x)

    ax2 = ax.twinx()
    b2 = ax2.bar(x + bar_width, Precision, width=bar_width, label='Precision', color=sns.xkcd_rgb["denim blue"])

    average_acc = sum(Accuracy)/len(Accuracy)
    average_prec = sum(Precision)/len(Precision)
    b3 = plt.hlines(y=average_acc, xmin=-bar_width, xmax=numclass - 1 + bar_width * 2, linewidth=2, linestyles='--', color='r',
               label='Average Acc : %0.2f' % average_acc)
    b4 = plt.hlines(y=average_prec, xmin=-bar_width, xmax=numclass - 1 + bar_width * 2, linewidth=2, linestyles='--', color='b',
               label='Average Prec : %0.2f' % average_prec)
    plt.xticks(np.arange(numclass) + bar_width / 2, np.arange(numclass))

    # labels, title and ticks
    ax.set_title('Accuracy and Precision Epoch #{}'.format(idx), fontsize=20)
    ax.set_xlabel('labels', fontsize=16)
    ax.set_ylabel('Acc(%)', fontsize=16)
    ax2.set_ylabel('Prec(%)', fontsize=16)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    ax.tick_params(axis='y', colors=b1[0].get_facecolor())
    ax2.tick_params(axis='y', colors=b2[0].get_facecolor())

    plt.legend(handles=[b1, b2, b3, b4])
    # fig.savefig(os.path.join(PREDICTIONS_PATH, "Accuracy-Precision_{}.png".format(idx)), dpi=fig.dpi)
    fig.savefig(os.path.join(PREDICTIONS_PATH, "Accuracy-Precision.png"), dpi=fig.dpi)

    plt.close()

    TopK_idx_acc = heapq.nlargest(top, range(len(Accuracy)), Accuracy.__getitem__)
    TopK_idx_prec = heapq.nlargest(top, range(len(Precision)), Precision.__getitem__)

    TopK_low_idx = heapq.nsmallest(top, range(len(Precision)), Precision.__getitem__)


    print('=' * 80)
    print('Accuracy Tok {0}: \n'.format(top))
    print('| Class ID \t Accuracy(%) \t Precision(%) |')
    for i in TopK_idx_acc:
        print('| {0} \t {1} \t {2} |'.format(i, round(Accuracy[i], 2), round(Precision[i], 2)))
    print('-' * 80)
    print('Precision Tok {0}: \n'.format(top))
    print('| Class ID \t Accuracy(%) \t Precision(%) |')
    for i in TopK_idx_prec:
        print('| {0} \t {1} \t {2} |'.format(i, round(Accuracy[i], 2), round(Precision[i], 2)))
    print('=' * 80)

    return TopK_low_idx

def EvaluateMetric(PREDICTIONS_PATH, train_results, idx):
    TopK_low_idx = plot_confusion_matrix(PREDICTIONS_PATH, train_results['grounds'], train_results['preds'], train_results['categories'], idx)
