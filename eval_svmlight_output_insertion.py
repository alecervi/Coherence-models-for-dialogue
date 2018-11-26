from __future__ import print_function

import logging, sys
from collections import defaultdict
import numpy as np
from optparse import OptionParser
from itertools import combinations
from random import random
# from sklearn.datasets import load_svmlight_file
from statsmodels.sandbox.stats.runs import mcnemar
from sklearn.metrics import accuracy_score, classification_report, average_precision_score

# parse commandline arguments
op = OptionParser()
op.add_option("--testfile",
              action="store", type=str, dest="testfile",
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
        label = map(lambda x: int(x)-1, label)
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
    return sum(precs) / len(precs)


def average_score(func, lists, *args):
    scores = []
    for _, candidates in lists.items():
        scores.append(func(candidates, *args))
    return sum(scores) / len(scores)


def read_test_file(path):
    query_id = None
    with open(path, 'r') as infile:
        for line in infile:
            if line[0] is not '#':
                yield query_id, line.strip().split()
            else:
                query_id = line.strip().split()[2]

def evaluate(testfile, predfile = None):
    queries = {}
    test_file = list(read_test_file(testfile))
    if predfile:
        with open(predfile, 'r') as pred:
            for prd, (doc_id, doc) in zip(pred, test_file):

                lbl, qid = doc[0], doc[1]
                # print('Lab: ', lbl, '  Qid: ', qid, ' Pred: ', prd)
                queries[doc_id] = queries.get(doc_id, {})
                queries[doc_id][qid] = queries[doc_id].get(qid, list())
                queries[doc_id][qid].append((float(prd.strip()), int(lbl)))
    else:
        for (doc_id, doc) in test_file:

                lbl, qid = doc[0], doc[1]
                # print('Lab: ', lbl, '  Qid: ', qid, ' Pred: ', prd)
                queries[doc_id] = queries.get(doc_id, {})
                queries[doc_id][qid] = queries[doc_id].get(qid, list())
                queries[doc_id][qid].append((random(), int(lbl)))

    # print('Testfile: ', len([i for i in read_test_file(testfile)]))
    # print('Predfile: ', len([i for i in read_test_file(predfile)]))

    # y_pred = list()
    # y_true = list()
    # for doc_id in queries:
    #     for qid in queries[doc_id]:
    #         pairs_numb = [i for i in combinations(queries[qid], 2) if i[0][1]!=i[1][1]]
    #         # print('Pairs numb: ', len(pairs_numb), ' Qid ',qid)
    #         for pair in pairs_numb:
    #             (pred_1, true_1), (pred_2, true_2) = pair
    #             y_pred.append(int(pred_1 <= pred_2))
    #             y_true.append(int(true_1 <= true_2))

    #
    # print("Accuracy: {:.4f}".format(accuracy_score(y_true, y_pred)))
    # print(classification_report(y_true, y_pred))
    print("\n Rank Metrics \n ")
    print("Average MAP: {:.4f}".format(average_score(map_score, queries)))
    print("Average MRR: {:.4f}".format(average_score(mrr_score, queries)))

    print("\n Precisions\n ")

    print("Average PREC@{}: {:.4f}".format(1, average_score(prec_at, queries, 1)))
    print("Average PREC@{}: {:.4f}".format(2, average_score(prec_at, queries, 2)))
    print("Average PREC@{}: {:.4f}".format(3, average_score(prec_at, queries, 3)))
    print("Average PREC@{}: {:.4f}".format(5, average_score(prec_at, queries, 5)))
    print("Average PREC@{}: {:.4f}".format(10, average_score(prec_at, queries, 10)))

    return


def test_mcnemar(predfile1, predfile2, testfile):
    pass


def is_interactive():
    return not hasattr(sys.modules['__main__'], '__file__')

argv = [] if is_interactive() else sys.argv[1:]
(opts, args) = op.parse_args(argv)


def main():
    if not opts.statsign:
        print('Evaluating...')
        evaluate(opts.testfile, opts.predfile)
    else:
        print('Performing Mc Nemar test ...')
        test_mcnemar(opts.predfile, opts.predfile2, opts.testfile)

if __name__ == '__main__':
    main()
