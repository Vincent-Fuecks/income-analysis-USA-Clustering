import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


class ModelAlteration():
    def strat_kfold_evaluation(
		self,
		df, 
		model, 
		target:int, 
		folds:int, 
		shuffle:bool=True, 
		random_state:int=None) -> [float, ([],[])]:
        '''
		Implements some centroid based clustering algorithms on n-dimensional data
			
		Parameters
		------------
		df			: Your dataframe
		model		: A scikitlearn model used to classify labels
		target 		: The index of your target column
		folds		: How often your dataframe should be split
		shuffle		: Specifies if the samples should be shuffled
		random_state: If shuffle=True, random_state specifies the used seed.
					if None, shuffle will always be random.
						
		Returns
		------------
		accuracy	: A list which contains the accuracy of the model over each folds
		best_fold	: The fold with the highest accuracy with the used model
	    '''
	
        data, target = df.loc[:, df.columns!=target].values, df[target].values      
        skf = StratifiedKFold(n_splits=folds, shuffle=shuffle, random_state=random_state)
        accuracy = [0 for _ in range(folds)]
        best_fold = []
        for i, index in enumerate(skf.split(data, target)):
            x_train, x_test = data[index[0]], data[index[1]]
            y_train, y_test = target[index[0]], target[index[1]]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_test)

            accuracy[i] = (100/ len(x_test)) * max(np.sum([1 if y_ == y else 0 for y_, y in zip(y_pred, y_test)]), 
                                                      np.sum([1 if y_ != y else 0 for y_, y in zip(y_pred, y_test)]))
            
            if accuracy[i] >= max(accuracy[:-1]): best_fold = index
        return(accuracy, best_fold)


    def plot_accuracy(self, acc:[[float]], xlab:str, legend:[str], xaxis:[]=[]):
        '''
        Plots all permutation of the parameters. 
        ------------
        acc         :[[float]]
                     Contains the accuracy of all folds.
        xlab        :String  
                     Contains the name for the x-axis.
        legend      :[String]
                     Contains the values for the plot legend.
        xaxis       :[int] or [float] 
                     Contains values for the x-axis.
        '''
        plt.xlabel(xlab)
        plt.ylabel('Accuracy [%]')
        acc = acc if len(acc)>0 else [acc]
        if not xaxis:
            for i, accuracy in enumerate(acc):
                plt.plot(range(len(accuracy)), accuracy, label = legend[i])
        else:
            for i, accuracy in enumerate(acc):
                plt.plot(xaxis, accuracy, label = legend[i])  
        plt.legend(loc="upper left")
        plt.show()


    def optimize_kMeans(self, 
        df, 
        target:int, 
        algorithm = ["full", "elkan"],
        max_iter = [1,2,5,7,10,20,50,100,200, 300, 500, 1000],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the optimal model parameters for the k-Means Clustering
        by finding the best fold for each permutation of the parameters. 
        The best fold is determined by strat_kfold_evaluation(). 
        The accuracy of all best folds is then compared and the  parameters of 
        the best fold are returned (in addition to the fold itself)
        Parameters
        ------------
        df          :dataframe
                     Your datatable.
        target      :int 
                     The index of your target column.
        algorithm   :[String]
                     A list which contains names of algorithm, wihch kMeans intern going to use.
        max_iter    :[int] 
                     Maximum number of iterations of the k-means algorithm for a single run.
        folds       :int 
                     How often your dataframe should be split in strat_kfold_evaluation.
        plot        :bool
                     Plots the accuracies over each fold.
        Returns
        ------------
        best_fold   :(np.array(int), {model_parameters})
                     An indexlist of the fold which has performed best overall, 
                    and a dict with the model parameters of the best fold 
        '''
       
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in max_iter] for _ in algorithm]
        epoch, end = 1, len(max_iter)*len(algorithm)
        for i,algo in enumerate(algorithm):
            for j, iter in enumerate(max_iter):
                model = KMeans(n_clusters = 2, max_iter = iter, algorithm = algo)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, {"n_clusters" : 2, "max_iter" : iter, "algorithm" : algo})
                print("Epoch %s/%s | max_iter =%s, algorithm=%s, Accuracy=%s" % (epoch, end, iter, algo, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Number of max_iter", list(map(lambda x: "algorithm " + x, list(algorithm))), max_iter)
        return(best_model)


    def optimize_kMedoids(self, 
        df, 
        target:int, 
        init = ['random', 'heuristic', 'k-medoids++', 'build'],
        max_iter = [100,200,300,400,500],
        folds:int = 10,
        plot:bool=True):
        '''
        Attempts to find the optimal model parameters for the k-Medoids Clustering
        by finding the best fold for each permutation of the parameters. 
        The best fold is determined by strat_kfold_evaluation(). 
        The accuracy of all best folds is then compared and the  parameters of 
        the best fold are returned (in addition to the fold itself)
        Parameters
        ------------
        df          :dataframe
                     Your datatable
        target      :int 
                     The index of your target column.
        init        :[String]
                     Specify medoid initialization method.
        max_iter     :[int] 
                     Specify the maximum number of iterations when fitting. 
        folds       :int 
                     How often your dataframe should be split in strat_kfold_evaluation.
        plot        :bool
                     Plots the accuracies over each fold.
        Returns
        ------------
        best_fold   :(np.array(int), {model_parameters})
                     An indexlist of the fold which has performed best overall, 
                    and a dict with the model parameters of the best fold 
        '''
       
        best_acc, best_model, fold_acc = 0, 0, [[None for _ in max_iter] for _ in init]
        epoch, end = 1, len(max_iter)*len(init)
        for i,v_init in enumerate(init):
            for j, iter in enumerate(max_iter):
                model = KMedoids(n_clusters = 2, max_iter = iter, init = v_init)
                fold_acc[i][j], tmp_fold = (lambda x: [max(x[0]), x[1]])(self.strat_kfold_evaluation(df, model, target, folds))
                if fold_acc[i][j] > best_acc: 
                    best_acc = fold_acc[i][j]
                    best_model = (tmp_fold, {"n_clusters" : 2, "max_iter" : iter, "init" : v_init})
                print("Epoch %s/%s | max_iter=%s, init=%s, Accuracy=%s" % (epoch, end, iter, v_init, fold_acc[i][j]))
                epoch += 1
        if plot: self.plot_accuracy(fold_acc, "Number of max_iter", list(map(lambda x: "init " + x, list(init))), max_iter)
        return(best_model)
