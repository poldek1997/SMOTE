# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:50:26 2020

@author: mdab
"""

import numpy as np
from scipy import sparse
from sklearn.utils import safe_indexing
from imblearn.utils import check_neighbors_object
from imblearn.over_sampling.base import BaseOverSampler
from sklearn.utils import check_random_state

class SMOTE(BaseOverSampler):
    
    def __init__(self,
                 sampling_strategy = 'auto',
                 random_state = None,
                 k_neighbors = 5,
                 n_jobs = 1):
        
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.k_neighbors = k_neighbors

    def _validate_estimator(self):
        self.nn_k = check_neighbors_object('K_neighbors',self.k_neighbors,additional_neighbor = 1)
        self.nn_k.set_params(**{'n_jobs': self.n_jobs})
       
    def _make_samples(self, X,y_dtype,y_type,nn_data,nn_num,n_samples,step_size=1.):

        random_state = check_random_state(self.random_state)

        samples_indices = random_state.randint(

            low=0, high=len(nn_num.flatten()), size=n_samples)

        steps = step_size * random_state.uniform(size=n_samples)

        rows = np.floor_divide(samples_indices, nn_num.shape[1])

        cols = np.mod(samples_indices, nn_num.shape[1])

        y_new = np.array([y_type] * len(samples_indices), dtype=y_dtype)

        if sparse.issparse(X):

            row_indices, col_indices, samples = [], [], []

            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):

                if X[row].nnz:

                    sample = self._generate_sample(X, nn_data, nn_num,row, col, step)

                    row_indices += [i] * len(sample.indices)

                    col_indices += sample.indices.tolist()

                    samples += sample.data.tolist()

            return (sparse.csr_matrix((samples, (row_indices, col_indices)),[len(samples_indices), X.shape[1]],dtype=X.dtype),y_new)

        else:

            X_new = np.zeros((n_samples, X.shape[1]), dtype=X.dtype)

            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):

                X_new[i] = self._generate_sample(X, nn_data, nn_num, row, col, step)

            return X_new, y_new

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        return X[row] - step * (X[row] - nn_data[nn_num[row, col]])

    def _fit_resample(self, X, y):
        self._validate_estimator()
        return self._sample(X,y)
    
    def _sample(self, X, y):
        X_resampled = X.copy()
        y_resampled = y.copy()
     
        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)
            
            self.nn_k.fit(X_class)
            nns = self.nn_k.kneighbors(X_class,return_distance=False)[:,1:]
            X_new, y_new = self._make_samples(X_class,y.dtype,class_sample,X_class,nns,n_samples,1.0)
            
            if sparse.issparse(X_new):

                X_resampled = sparse.vstack([X_resampled, X_new])

                sparse_func = 'tocsc' if X.format == 'csc' else 'tocsr'

                X_resampled = getattr(X_resampled, sparse_func)()

            else:

                X_resampled = np.vstack((X_resampled, X_new))

            y_resampled = np.hstack((y_resampled, y_new))
  
        return X_resampled, y_resampled
        
 
            