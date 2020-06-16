import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler,ADASYN
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from SMOTE import SMOTE
from imblearn.metrics import geometric_mean_score, specificity_score, sensitivity_score
from tabulate import tabulate
from scipy.stats import ranksums
from scipy.stats import rankdata

dataset = ['zoo-3','glass-0-1-5_vs_2','ecoli-0-1-4-6_vs_5','ecoli4','glass4','page-blocks-1-3_vs_4','abalone19','poker-9_vs_7','yeast5','dermatology-6']
   
preprocs = {
    'none': None,
    'ros': RandomOverSampler(random_state=1410),
    'rus': RandomUnderSampler(random_state=1410),
    'smote' : SMOTE(random_state=1410),
    'adasyn': ADASYN(random_state=1410),
}

datasettry = {
    'zoo-3':None,
    'glass-0-1-5_vs_2':None,
    'ecoli-0-1-4-6_vs_5': None,
    'ecoli4': None,
    'glass4': None,
    'page-blocks-1-3_vs_4': None,
    'abalone19':None,
    'poker-9_vs_7':None,
    'yeast5':None,
    'dermatology-6':None
}

metrics = {
    'g-mean': geometric_mean_score,
    'specificity': specificity_score,
    'recall': sensitivity_score,
}

n_splits = 5
n_repeats = 2
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=1234)

scores = np.zeros((len(preprocs),len(dataset),n_splits * n_repeats,len(metrics)))
    

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
            for metric_id, metric in enumerate(metrics):
                scores[preproc_id,data_id, fold_id, metric_id] = metrics[metric](y[test], y_pred)

            
mean_scores = np.mean(scores, axis=3)
mean_scores = np.mean(mean_scores, axis=2).T


ranks = []
for ms in mean_scores:
    ranks.append(rankdata(ms).tolist())
ranks = np.array(ranks)


mean_ranks = np.mean(ranks, axis=0)

alfa = .05
w_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        w_statistic[i, j], p_value[i, j] = ranksums(ranks.T[i], ranks.T[j])


headers = list(preprocs.keys())
names_column = np.expand_dims(np.array(list(preprocs.keys())), axis=1)
w_statistic_table = np.concatenate((names_column, w_statistic), axis=1)
w_statistic_table = tabulate(w_statistic_table, headers, floatfmt=".2f")
p_value_table = np.concatenate((names_column, p_value), axis=1)
p_value_table = tabulate(p_value_table, headers, floatfmt=".2f")

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[w_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)


names_column1 = np.expand_dims(np.array(list(datasettry.keys())), axis=1)
m = np.concatenate((names_column1, mean_scores), axis=1)
m = tabulate(m, headers, tablefmt=".2f")


print("\n\n",m)
print("\n")
print("Statistical significance (alpha = 0.05): \n", significance_table)
print("\n")

from scipy.stats import ttest_ind

mean_scores1 = np.mean(scores, axis=3)
mean_scores1 = np.mean(mean_scores1, axis=1).T

t_statistic = np.zeros((len(preprocs), len(preprocs)))
p_value = np.zeros((len(preprocs), len(preprocs)))

for i in range(len(preprocs)):
    for j in range(len(preprocs)):
        t_statistic[i, j], p_value[i, j] = ttest_ind(mean_scores1[i], mean_scores1[j])

advantage = np.zeros((len(preprocs), len(preprocs)))
advantage[t_statistic > 0] = 1
advantage_table = tabulate(np.concatenate(
    (names_column, advantage), axis=1), headers)

significance = np.zeros((len(preprocs), len(preprocs)))
significance[p_value <= alfa] = 1
significance_table = tabulate(np.concatenate(
    (names_column, significance), axis=1), headers)

stat_better = significance * advantage
stat_better_table = tabulate(np.concatenate(
    (names_column, stat_better), axis=1), headers)

print("Statistically significantly better:\n", stat_better_table)

print("\n")







            
            
