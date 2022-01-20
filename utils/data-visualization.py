import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.express as px

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

class data_visualization():
    def prepare_data(self, df, best_model, n_comp = 2):
        '''
        Visualization of decision boundaries from the classifiers.
        A dimension reduction to two features is applied, to visualize the 
        boundaries. 
        ------------
        df          : dataframe
                     Your datatable
        best_model  :(np.array(int))
                     An indexlist of the fold which has performed best overall (only need this)
        Returns
        ------------
        features  :(np.array(float)) X_train of the best performing model
        y         :(np.array(float)) y_train of the best performing model
        '''
        arr = np.array(df.iloc[best_model[0]])
        features = self.dimension_reduction_to_n_dimension(arr[:, :-1], n_comp)
        return features, y = arr[:, -1]




    def visualization(self, df, best_model, model):
        '''
        Visualization of decision boundaries from the classifiers.
        A dimension reduction to two features is applied, to visualize the 
        boundaries. 
        ------------
        df          : dataframe
                    Your datatable
        best_model  :(np.array(int))
                    An indexlist of the fold which has performed best overall (only need this)
        model:      is classifier with the parameters, of the best fold 
        '''
        f_redu, y =  self.prepare_data(df, best_model)
        self.plot_decision_boundary2D(f_redu, y, model)
        print("\n")
        f_redu, y =  self.prepare_data(df, best_model, 3)
        self.plot_3D(f_redu, y, model)

    def plot_3D(self, x_data, y_data, model):
        """
        Fist plot reducd best fold data, Secound plot best fold clustered with best parameters in 3D
        ------------
        x_data      :(np.array(float))  X_train for the best fold 
        y_data      :(np.array(float)) target data of the feature data  
        model       : scikit learn model 
        """
        df = pd.DataFrame(data = np.c_[x_data, y_data], columns=['x', 'y', 'z', 'color'])
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='color', 
                             labels={
                                    "x": "feature 0",
                                    "y": "feature 1",
                                    "z": "feature 2" },
                            title="Adult dataset best fold reduced to 3D")
        fig.show()
        

        arr_0 = np.array(df)
        features_0 = self.dimension_reduction_to_n_dimension(arr_0[:, :-1], 3)
        y_0 = arr_0[:, -1]
        model.fit(features_0)
        y_pred = model.predict(x_data)
        df = pd.DataFrame(data = np.c_[x_data, y_pred], columns=['x', 'y', 'z', 'color'])
        print("\n")
        fig = px.scatter_3d(df, x='x', y='y', z='z', color='color',
                            labels={
                                    "x": "feature 0",
                                    "y": "feature 1",
                                    "z": "feature 2" },
                            title= "Adult dataset best fold reduced to 3D, clustered with " + str(model) )
        fig.show()
        
        
    def dimension_reduction_to_n_dimension(self, data, n_comp = 2):
        """
        Performs a dimension reduction to 2 components. Use StandardScaler, for 
        a nicer visualization. For the reduction is used PCA. 
        ------------
        df                  : (np.array(float)), data for dimension reduction 
        Returns
        ------------
        principalComponents : (np.array(float)) data of only to features 
        """
        x = StandardScaler().fit_transform(data)
        pca = PCA(n_components= n_comp)
        principalComponents = pca.fit_transform(x)
        return principalComponents
    

    def plot_decision_boundary2D(self,features, target, model):
        """
        Plots the feature points and draws the decision boundaries from the classifier.
        We used part of the code of the Website for this function:
        https://scikit-learn.org/stable/auto_examples/tree/plot_iris_dtc.html
        ------------
        features    :(np.array(float))  X_train for the best fold 
        target      :(np.array(float)) target data of the feature data  
        model       : scikit learn model 
        """
        y = np.array(target).flatten()
        model.fit(features, target)

       

        # Step size of the mesh
        h = .01   

        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        

        x_min, x_max = features[:, 0].min() - 1, features[:, 0].max() + 1
        y_min, y_max = features[:, 1].min() - 1, features[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                            np.arange(y_min, y_max, 0.1))

        # Predictions to obtain the classification results
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Plotting
        plt.title(model)
        colors = ['orange', 'indigo']
        co = [colors[x] for x in y.astype(int)]
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.scatter(features[:, 0], features[:, 1], c= co, alpha = 0.1)
        red_patch = mpatches.Patch(color='orange', label="Income <= 50K")
        blue_patch = mpatches.Patch(color='indigo', label="Income > 50K")
        plt.legend(handles=[red_patch, blue_patch])
        plt.xlabel("Feature-1")
        plt.ylabel("Feature-2")
        plt.show()
