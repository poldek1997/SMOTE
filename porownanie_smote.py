# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 14:19:57 2020

@author: mdab
"""


import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler,ADASYN
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from SMOTE import SMOTE

dataset = ['ecoli-0-1-4-6_vs_5','ecoli-0-3-4-7_vs_5-6','australian','diabetes']
   
preprocs = {
    'ros': RandomOverSampler(random_state=1410),
    'rus': RandomUnderSampler(random_state=1410),
    'smote' : SMOTE(random_state=1410),
    'adasyn': ADASYN(random_state=1410),
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(preprocs),len(dataset),n_splits * n_repeats))
    

for data_id, dataset in enumerate(dataset):
    dataset = np.genfromtxt("%s.csv" % (dataset), delimiter=",")
    X = dataset[:, :-1]
    y = dataset[:, -1].astype(int)
    
    for fold_id, (train, test) in enumerate(rskf.split(X, y)):
        for preproc_id, preproc in enumerate(preprocs):
            clf = GaussianNB()
    
            if preprocs[preproc] == None:
                X_train, y_train = X[train], y[train]
            else:
                X_train, y_train = preprocs[preproc].fit_resample(
                    X[train], y[train])
    
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X[test])
            scores[preproc_id, data_id, fold_id] = accuracy_score(y[test], y_pred)

            
mean_scores = np.mean(scores, axis=2).T


from scipy.stats import rankdata
ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)


mean_ranks = np.mean(ranks, axis=0)

from scipy.stats import ranksums

alfa = .05
w_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])

from tabulate import tabulate

headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")
print("\nw-statistic:\n", w_statistic_table, "\n\np-value:\n", p_value_table)

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)
print("\nAdvantage:\n", advantage_table)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)
print("\nStatistical significance (alpha = 0.05):\n", significance_table)







            
            