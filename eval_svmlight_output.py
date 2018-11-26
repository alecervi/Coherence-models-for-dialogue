from __future__ import print_function

import logging, sys
from collections import defaultdict
import numpy as np
from optparse import OptionParser
from itertools import combinations
from random import random
# from statsmodels.sandbox.stats.runs import mcnemar
from sklearn.metrics import confusion_matrix
from mlxtend.evaluate import mcnemar_table
from statsmodels.stats.contingency_tables import mcnemar
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar as mlx_mcnemar
from scipy.stats import mannwhitneyu, wilcoxon
# from sklearn.datasets import load_svmlight_file
from sklearn.metrics import accuracy_score, classification_report, average_precision_score

# parse commandline arguments
op = OptionParser()
op.add_option("--testfile",
              action="store", type=str, dest="testfile",
              help="Testfile.")
op.add_option("--testfile2",
              action="store", type=str, dest="testfile2",
              help="Testfile.")
op.add_option("--predfile",
              action="store", type=str, dest="predfile",
              help="Prediction file.")
op.add_option("--predfile2",
              action="store", type=str, dest="predfile2",
              help="Prediction file to compare with.")
op.add_option("--statsign",
              action="store_true", dest="statsign",
              help="Prediction file to compare with.")


def map_score(lists):
    average_precs = []
    for _, candidates in lists.items():
        score, label = zip(*candidates)
        label = list(map(lambda x: int(x)-1, label))
        average_precs.append(average_precision_score(label, score))
    return sum(average_precs) / len(average_precs)

def mrr_score(lists):
    recp_ranks = []
    for _, candidates in lists.items():
        rank = 0
        for i, (_, label) in enumerate(sorted(candidates, reverse=True, key=lambda x: x[0]), 1):
            if label == 2:
                rank += 1. / i
                break
        recp_ranks.append(rank)
    return sum(recp_ranks) / len(recp_ranks)

def prec_at(lists, n):
    precs = []
    for _, candidates in lists.items():
        for i, (_, label) in enumerate(sorted(candidates, reverse=True, key=lambda x: x[0]), 1):
            if i > n:
                precs.append(0.)
                break
            elif label == 2:
                precs.append(1.)
                break
    return sum(precs) / len(precs), precs


def read_test_file(path):
    with open(path, 'r') as infile:
        print('Reading: ', path)
        for line in infile:
            if line[0] is not '#':
                yield line.strip().split()

def evaluate(testfile, predfile = None):
    print('Predfile: ', predfile)
    print('Testfile: ', testfile)
    queries = defaultdict(list)
    test_file = list(read_test_file(testfile))
    if predfile:
        with open(predfile, 'r') as pred:
            for prd, doc in zip(pred, test_file):

                lbl, qid = doc[0], doc[1]
                # print('Lab: ', lbl, '  Qid: ', qid, ' Pred: ', prd)
                queries[qid].append((float(prd.strip()), int(lbl)))
    else:
        for doc in test_file:

                lbl, qid = doc[0], doc[1]
                # print('Lab: ', lbl, '  Qid: ', qid, ' Pred: ', prd)
                queries[qid].append((random(), int(lbl)))

    # print('Testfile: ', len([i for i in read_test_file(testfile)]))
    # print('Predfile: ', len([i for i in read_test_file(predfile)]))

    y_pred = list()
    y_true = list()
    for qid in queries:
        pairs_numb = [i for i in combinations(queries[qid], 2) if i[0][1]!=i[1][1]]
        # print('Pairs numb: ', len(pairs_numb), ' Qid ',qid)
        for pair in pairs_numb:
            (pred_1, true_1), (pred_2, true_2) = pair
            y_pred.append(int(pred_1 <= pred_2))
            y_true.append(int(true_1 <= true_2))


    print("Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred))
    print("\n Rank Metrics \n ")
    print("MAP: {:.4f}".format(map_score(queries)))
    print("MRR: {:.4f}".format(mrr_score(queries)))

    print("\n Precisions\n ")

    print("PREC@{}: {:.4f}".format(1, prec_at(queries, 1)[0]))
    print("PREC@{}: {:.4f}".format(2, prec_at(queries, 2)[0]))
    print("PREC@{}: {:.4f}".format(3, prec_at(queries, 3)[0]))
    print("PREC@{}: {:.4f}".format(5, prec_at(queries, 5)[0]))
    print("PREC@{}: {:.4f}".format(10, prec_at(queries, 10)[0]))
    y_pred_prec = prec_at(queries, 1)[1]
    y_true_prec = np.ones(len(y_pred_prec))

    return y_true, y_pred, y_true_prec, y_pred_prec


def test_mannwhithney(predfile1, predfile2, testfile, testfile2):
    y_true1, y_pred1, y_true_prec1, y_pred_prec1 = evaluate(testfile, predfile1)
    y_true2, y_pred2, y_true_prec2, y_pred_prec2 = evaluate(testfile2, predfile2)
    print('\n First model: ', predfile1)
    print('Ex: ', y_pred1[:10], ' Len: ', len(y_pred1))
    print('Second model: ', predfile2)
    print('Ex: ', y_pred2[:10], ' Len: ', len(y_pred2))
    print('Is testset the same? ', len([i for i in np.equal(np.array(y_true1), np.array(y_true2)) if i is False]))

    mc_tb = mcnemar_table(y_target=np.array(y_true1),
                       y_model1=np.array(y_pred1),
                       y_model2=np.array(y_pred2))
    print('Contingency table: ', mc_tb)
    mcnemar_res = mcnemar(mc_tb)
    print('McNemar:  p value: {:.20f}'.format(mcnemar_res.pvalue))
    chi2, p = mlx_mcnemar(ary=mc_tb, corrected=True)
    print('McNemar: chi:{:.4f}  p value: {}'.format(chi2, p))
    mc_tb_prec = mcnemar_table(y_target=np.array(y_true_prec1),
                                y_model1=np.array(y_pred_prec1),
                                y_model2=np.array(y_pred_prec2))
    mcnemar_res_prec = mcnemar(mc_tb_prec)
    print('McNemar PRECISION:  p value: {}'.format(mcnemar_res_prec.pvalue))
    # mw_stat, mw_p_val = mannwhitneyu(np.array(y_pred1), np.array(y_pred2), alternative='less')
    # print('Mann Whitney: Stats: ', mw_stat, ' p value: ', mw_p_val)
    # wil_stat, wil_p_val = wilcoxon(np.array(y_pred1), np.array(y_pred2))
    # print('Wilcoxon: Stats: ', wil_stat, ' p value: ', wil_p_val)

def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)


def main():
    if not opts.statsign:
        print('Evaluating...')
        evaluate(opts.testfile, opts.predfile)
    else:
        print('Performing Mann Whitney test ...')
        test_mannwhithney(opts.predfile, opts.predfile2, opts.testfile, opts.testfile2)

if __name__ == '__main__':
    main()
