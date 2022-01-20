import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


class evaluation_measures():
    def clustering_results(self, df, model):
        '''
        This function clusters the data with given model and returns 
        the data of each cluster in an array.

        Parameters
        ------------
        df        : DataFrame
                    Contains data for clustering.
        model     :Is a sklearn clustering model.  
        Returns
        ------------
        c_result  : Return in a tupel the data of each cluster in an array.
        '''        
        X_train = np.array(df)[:-1]
        y_pred = model.fit(X_train).labels_
        return (X_train[y_pred], X_train[0 == y_pred])
        

    def cluster_purity(self, c_result):
        '''
        This function calculates for each cluster the purity.
        Parameters
        ------------
        c_result  : Is a tupel, which contains the separated data of each cluster
        '''    
        all_purity = ([None]*len(c_result))
        for i, C in enumerate(c_result):
            tmp = np.sum(C, axis = 0)
            all_purity[i] =  (1/np.sum(tmp)) * np.max(tmp)
            print("Purity for Cluster %d is %f" %(i, all_purity[i]))
        print("Purity is %f \n" %( np.sum(all_purity)))



    def cluster_entropy(self, c_result):
        '''
        This function calculates for each cluster the entropy.
        Parameters
        ------------
        c_result  : Is a tupel, which contains the separated data of each cluster
        '''    
        all_e = np.array([None]*len(c_result))
        for i, C in enumerate(c_result):
            m = np.sum(C, axis = 0)
            e_j = 0
            for m_j in m: 
                log = 0 if m_j == 0 or np.sum(m) == 0 else np.log(m_j / np.sum(m))   
                e_j += m_j/np.sum(m) * log
            all_e[i] = e_j*-1
            print("Entropy for Cluster %d is %f" %(i, all_e[i]))
        print("Purity is %f \n" %(np.sum(all_e)))
