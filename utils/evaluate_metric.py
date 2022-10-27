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
# Plot accuracy and loss
# ---------------------------------------
def get_error_bar(best_score, valid_examples):
    print("--------------------------------------------")
    print("Standard Error") # best_score: Average al of scores, valid_examples: num of all samples
    print("--------------------------------------------")

    err = np.sqrt((best_score * (1 - best_score)) / valid_examples)
    err_rounded_68 = round(err, 2)
    err_rounded_95 = round((err_rounded_68 * 2), 2)

    print('Error (68% CI): +- ' + str(err_rounded_68))
    print('Error (95% CI): +- ' + str(err_rounded_95))
    print()
    return err_rounded_68

def plot_train_results(PREDICTIONS_PATH, train_results, idx):
    '''

    :param PREDICTIONS_PATH: plot image save path
    :param train_results: {'valid_score':[...], 'valid_loss':[...], 'train_score': [...], 'train_loss':[...]}
    :param best_score: validation best acc
    :param idx: epoch index
    :return: None
    '''

    # best_score = sum(train_results['valid_score']) / len(train_results['valid_score'])
    valid_examples = len(train_results['valid_score'])
    super_category = str(idx)

    best_score = train_results["best_score"]
    standard_error = get_error_bar(best_score, valid_examples)
    y_upper = train_results["valid_score"] + standard_error
    y_lower = train_results["valid_score"] - standard_error

    print("--------------------------------------------")
    print("Results")
    print("--------------------------------------------")

    fig = plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, len(train_results["train_score"])), train_results["train_score"], label='train')

    plt.plot(range(0, len(train_results["valid_score"])), train_results["valid_score"], label='valid')

    kwargs = {'color': 'black', 'linewidth': 1, 'linestyle': '--', 'dashes': (5, 5)}
    plt.plot(range(0, len(train_results["valid_score"])), y_lower, **kwargs)
    plt.plot(range(0, len(train_results["valid_score"])), y_upper, **kwargs, label='validation SE (68% CI)')

    plt.title('Accuracy Plot - ' + super_category, fontsize=20)
    plt.ylabel('Accuracy', fontsize=16)
    plt.xlabel('Training Epochs', fontsize=16)
    plt.ylim(0, 1)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, len(train_results["train_loss"])), train_results["train_loss"], label='train')
    plt.plot(range(0, len(train_results["valid_loss"])), train_results["valid_loss"], label='valid')

    plt.title('Loss Plot - ' + super_category, fontsize=20)
    plt.ylabel('Loss', fontsize=16)
    plt.xlabel('Training Epochs', fontsize=16)
    max_train_loss = max(train_results["train_loss"])
    max_valid_loss = max(train_results["valid_loss"])
    y_max_t_v = max_valid_loss if max_valid_loss > max_train_loss else max_train_loss
    ylim_loss = y_max_t_v if y_max_t_v > 1 else 1
    plt.ylim(0, ylim_loss)
    plt.legend()

    plt.show()

    fig.savefig(os.path.join(PREDICTIONS_PATH, "train_results_{}.png".format(idx)), dpi=fig.dpi)


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


# Fast Rank Pooling
sample_size = 128
def GenerateRPImage(imgs_path, sl):
    def get_DDI(video_arr):
        def get_w(N):
            return [float(i) * 2 - N - 1 for i in range(1, N + 1)]

        w_arr = get_w(len(video_arr))
        re = np.zeros((sample_size, sample_size, 3))
        for a, b in zip(video_arr, w_arr):
            img = cv2.imread(os.path.join(imgs_path, "%06d.jpg" % a))
            img = cv2.resize(img, (sample_size, sample_size))
            re += img * b
        re -= np.min(re)
        re = 255.0 * re / np.max(re) if np.max(re) != 0 else 255.0 * re / (np.max(re) + 0.00001)

        return re.astype('uint8')

    return get_DDI(sl)

# ---------------------------------------
# Wrongly Classified Images
# ---------------------------------------
def plot_wrongly_classified_images(PREDICTIONS_PATH, TopK_low_idx, valid_images, idx):
    print("--------------------------------------------")
    print("Wrongly Classified Images")
    print("--------------------------------------------")

    v_paths, grounds, preds = valid_images
    f = lambda n, sn: [(lambda n, arr: n if arr == [] else int(np.mean(arr)))(n * i / sn, range(int(n * i / sn),
                                                                                            max(int(
                                                                                                n * i / sn) + 1,
                                                                                                int(n * (
                                                                                                        i + 1) / sn))))
                   for i in range(sn)]

    train_images = []
    ground, pred, pred_lbl_file = [], [], []
    for g, p, v in zip(grounds, preds, v_paths):
        assert p != g, 'Pred: {} equ to ground-truth: {}'.format(p, g)
        if g in TopK_low_idx[:10]:
            imgs = [transforms.ToTensor()(Image.open(os.path.join(v, "%06d.jpg" % a)).resize((200, 200))).unsqueeze(0) for a in f(len(os.listdir(v))//2, 10)]
            train_images.append(make_grid(torch.cat(imgs), nrow=10, padding=2).permute(1, 2, 0))
            ground.append(g)
            pred.append(p)
            pred_lbl_file.append(v)
        if len(train_images) > 9:
            break

    fig = plt.figure(figsize=(30, 20))
    k = 0
    for i in range(0, len(train_images)):
        fig.add_subplot(10, 1, k + 1)
        plt.axis('off')
        if i == 0:
            title = "Orig lbl: " + str(ground[i]) + " Pred lbl: " + str(pred[i]) + "  " + pred_lbl_file[i]
        else:
            title = '\n'*10 + "Orig lbl: " + str(ground[i]) + " Pred lbl: " + str(pred[i]) + "  " + pred_lbl_file[i]
        plt.title(title)
        plt.imshow(train_images[i])
        k += 1

    plt.pause(0.1)
    print()
    fig.savefig(os.path.join(PREDICTIONS_PATH, "wrongly_classified_images.png".format(idx)), dpi=fig.dpi)
    plt.close()

def EvaluateMetric(PREDICTIONS_PATH, train_results, idx):
    TopK_low_idx = plot_confusion_matrix(PREDICTIONS_PATH, train_results['grounds'], train_results['preds'], train_results['categories'], idx)
