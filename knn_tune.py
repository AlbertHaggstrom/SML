
from dataprep import getdata_minmax_mean
from tester import tester
import sklearn.neighbors as skl_nb
from sklearn.model_selection import GridSearchCV

def knn_tuning(features,label):
    param_grid = {
        'n_neighbors':range(1,20),
        'weights': ['uniform', 'distance'],
        'algorithm': ['brute'],
        'metric': ['minkowski', 'cosine'],
        'p': range(1,11)
        }
 
    model = skl_nb.KNeighborsClassifier()
    grid = GridSearchCV(model,param_grid=param_grid,verbose=1,n_jobs=-1,scoring='balanced_accuracy')
    grid_fit = grid.fit(features,label)
    
    return grid_fit



if __name__ == '__main__':
    features = ['Perc NMA', 'Perc NFA', 'AL/AF', 'ACL/AF', 'Perc WF', 'Perc WM', 'Perc WCL', 'WpM/WCL', 'WpF/WCL', 'WpM/WpA', 'WpF/WpA']
    x_train, x_test, y_train, y_test = getdata_minmax_mean()
    grid = knn_tuning(x_train[features],y_train)
    
 
    print('\n Best estimator:\n', grid.best_estimator_)
    print(grid.best_score_ * 2 - 1)
    print('\n Best parameters:', grid.best_params_)
    model = grid.best_estimator_
    print("\n Scores on y_test:", tester(model,x_test[features],y_test))

    

