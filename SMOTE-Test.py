# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:30:49 2020

@author: mdab
"""


from collections import Counter
from sklearn.datasets import make_classification
from SMOTE import SMOTE


X, y = make_classification(n_classes=2, class_sep=2,
weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
n_features=25, n_clusters_per_class=1, n_samples=10000, random_state=10)
print('Original dataset shape %s' % Counter(y))

sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)
print('Resampled dataset shape %s' % Counter(y_res))
