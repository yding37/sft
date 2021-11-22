import torch
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F

from utils.common import *
from datasets.mosei_dataset import *
from utils.vggsound_eval import calculate_stats, d_prime, topk_accuracies


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth

    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def weighted_accuracy(test_preds_emo, test_truth_emo):
    true_label = (test_truth_emo > 0)
    predicted_label = (test_preds_emo > 0)
    tp = float(np.sum((true_label == 1) & (predicted_label == 1)))
    tn = float(np.sum((true_label == 0) & (predicted_label == 0)))
    p = float(np.sum(true_label == 1))
    n = float(np.sum(true_label == 0))

    return (tp * (n/p) + tn) / (2*n)

def eval_mosei_senti(results, truths, exclude_zero=False):
    labels = truths.cpu()
    labels_oh = F.one_hot(labels, 7).cpu()
    preds = results.cpu()

    stats = calculate_stats(preds, labels_oh)

    tk_list = [1, 3]
    tk = topk_accuracies(preds, labels, tk_list)

    _, top_max_k_inds = torch.topk(
        preds, 1, dim=1, largest=True, sorted=True
    )
    simple_preds = []
    for i in top_max_k_inds:
        if i >= 3:
            simple_preds.append(1)
        elif i < 3:
            simple_preds.append(-1)
        # else:
        #     simple_preds.append(0)

    simple_truths = []
    for i in labels:
        if i > 3:
            simple_truths.append(1)
        elif i < 3:
            simple_truths.append(-1)
        else:
            simple_truths.append(0)

    # print(simple_preds, simple_truths)

    simple_stats = eval_mosei_senti_original(simple_preds, simple_truths, exclude_zero=True)
    print(simple_stats)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    dprime = d_prime(mAUC)

    return {
        'top1': tk[0].numpy().item(),
        'top3': tk[1].numpy().item(),
        'mAP': mAP,
        'mAUC': mAUC,
        'dprime': dprime,
        'f1': simple_stats['f1'],
        'acc2': simple_stats['acc2']
    }

def eval_mosei_senti_original(results, truths, exclude_zero=False):
    test_preds = np.array(results)#.view(-1).cpu().detach().numpy()
    test_truth = np.array(truths)#.view(-1).cpu().detach().numpy()

    # print(test_preds, test_truth)
    non_zeros = np.array([i for i, e in enumerate(
        test_truth) if e != 0 or (not exclude_zero)])

    test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
    test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)
    test_preds_a5 = np.clip(test_preds, a_min=-2., a_max=2.)
    test_truth_a5 = np.clip(test_truth, a_min=-2., a_max=2.)

    # Average L1 distance between preds and truths
    mae = np.mean(np.absolute(test_preds - test_truth))
    corr = np.corrcoef(test_preds, test_truth)[0][1]
    mult_a7 = multiclass_acc(test_preds_a7, test_truth_a7)
    mult_a5 = multiclass_acc(test_preds_a5, test_truth_a5)
    f_score = f1_score((test_preds[non_zeros] > 0),
                       (test_truth[non_zeros] > 0), average='weighted')
    binary_truth = (test_truth[non_zeros] > 0)
    binary_preds = (test_preds[non_zeros] > 0)

    logging.info("MAE: %f", mae)
    logging.info("Correlation Coefficient: %f", corr)
    logging.info("mult_acc_7: %f", mult_a7)
    logging.info("mult_acc_5: %f", mult_a5)
    logging.info("F1 score: %f", f_score)
    mult_a2 = accuracy_score(binary_truth, binary_preds)
    logging.info("Accuracy: %f", mult_a2)

    logging.info("-" * 50)

    return {
        'mae': mae,
        'corr': corr,
        'acc7': mult_a7,
        'acc5': mult_a5,
        'f1': f_score,
        'acc2': mult_a2
    }


def eval_mosi(results, truths, exclude_zero=False):
    return eval_mosei_senti(results, truths, exclude_zero)


def eval_iemocap(results, truths, single=-1):
    emos = ["Neutral", "Happy", "Sad", "Angry"]
    if single < 0:
        test_preds = results.view(-1, 4, 2).cpu().detach().numpy()
        test_truth = truths.view(-1, 4).cpu().detach().numpy()

        for emo_ind in range(4):
            print(f"{emos[emo_ind]}: ")
            test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
            test_truth_i = test_truth[:, emo_ind]
            f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
            acc = accuracy_score(test_truth_i, test_preds_i)
            print("  - F1 Score: ", f1)
            print("  - Accuracy: ", acc)
    else:
        test_preds = results.view(-1, 2).cpu().detach().numpy()
        test_truth = truths.view(-1).cpu().detach().numpy()

        print(f"{emos[single]}: ")
        test_preds_i = np.argmax(test_preds, axis=1)
        test_truth_i = test_truth
        f1 = f1_score(test_truth_i, test_preds_i, average='weighted')
        acc = accuracy_score(test_truth_i, test_preds_i)
        print("  - F1 Score: ", f1)
        print("  - Accuracy: ", acc)
