import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


class evaluation_measures():
    def clustering_results(self, df, best_model, model):
        '''
        This function clusters the data with given model and returns 
        the data of each cluster in an array.

        Parameters
        ------------
        df        : DataFrame
                    Contains data for clustering.
        best_model: np.array([]) Has index of best fold
        model     : Is a sklearn clustering model.  
        Returns
        ------------
        c_result  : Return in a tupel the data of each cluster in an array.
        '''        
        data = np.array(df.iloc[best_model[0]])
        features = data[:, :-1]
        target = data[:, -1]
        y_pred = model.fit(features).predict(features)
        total = target.shape[0]
        cluster_0 = np.sum([y_t == y and y ==1  for y_t, y in zip(target, y_pred)])
        cluster_1 = [y_t == y and y == 0 for y_t, y in zip(target, y_pred)]
        return cluster_0, cluster_1, total
        

    def cluster_purity(self, c_result):
        '''
        This function calculates for each cluster the purity.
        Parameters
        ------------
        c_result  : Is a tupel, which contains cluster_0, cluster_0, total 
        '''    
        cluster_0, cluster_1, total = c_result
        return (np.sum(cluster_0) + np.sum(cluster_1))/ total



    def cluster_entropy(self, c_result):
        '''
        This function calculates for each cluster the entropy.
        Parameters
        ------------
        c_result  : Is a tupel, which contains cluster_0, cluster_0, total 
        '''    
        cluster_0, cluster_1, total = c_result
        return -((np.sum(cluster_0)/total) * np.log((np.sum(cluster_0)/total) )) - ((np.sum(cluster_1)/total) * np.log((np.sum(cluster_1)/total) ))
