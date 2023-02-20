from dataprep import getdata_minmax_mean, get_testset
from tester import tester
from save import save_predictions
import sklearn.neighbors as skl_nb


def knn_algorithm(features, label):
    model = skl_nb.KNeighborsClassifier(n_neighbors=1, weights='uniform', metric = 'cosine', algorithm='brute')
    model.fit(features, label)
    return model


if __name__ == '__main__':
    x_train, x_test, y_train, y_test= getdata_minmax_mean()
    features = ['Perc NMA', 'Perc NFA', 'AL/AF', 'ACL/AF', 'Perc WF', 'Perc WM', 'Perc WCL', 'WpM/WCL', 'WpF/WCL', 'WpM/WpA', 'WpF/WpA']    
    model = knn_algorithm(x_train[features],y_train)
    
    print("\n Scores on y_test:", tester(model,x_test[features],y_test))
    
    x_eval = get_testset()
    predict = model.predict(x_eval[features])
    save_predictions(predict)
    
    