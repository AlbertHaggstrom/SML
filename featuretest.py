from dataprep import getdata_minmax_mean
from sklearn.model_selection import StratifiedKFold
from itertools import chain, combinations
from knn import knn_algorithm
import matplotlib.pyplot as plt
from tester import tester
import time

def powerset(col_names):
    #Returns the powerset of all the column names (all combinations).
    s = list(col_names)
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)) #range(1,... is without the empty set

if __name__ == '__main__':
    #Brute force feature selection, 5 folds CV.
    start = time.time()
    x_train, x_test, y_train, y_test = getdata_minmax_mean()
    
    skf = StratifiedKFold(n_splits=5)
    list_combos_acc = []
    list_combos_bal_acc = []

    for fold, (train_i, test_i) in enumerate(skf.split(x_train, y_train)):
        fx_train = x_train.iloc[train_i]
        fy_train = y_train.iloc[train_i]
        fx_test = x_train.iloc[test_i]
        fy_test = y_train.iloc[test_i]

        col_names = list(x_train.columns)
        feature_combos = list(powerset(col_names))
        acc = []
        bal_acc = []

        for i in range(len(feature_combos)): # check accuracy for all the combos on the column name powerset
            model = knn_algorithm(fx_train[list(feature_combos[i])], fy_train)
            pred = model.predict(fx_test[list(feature_combos[i])])
            ac, bac = tester(model,fx_test[list(feature_combos[i])], fy_test)
            acc.append(ac)
            bal_acc.append(bac)
   
        combos_acc = zip(feature_combos, acc)  #combine accuracies and feature combos
        combos_acc_sorted = sorted(combos_acc, key=lambda x: x[1], reverse=True) #sort on highest accuracy first
        top_combos_acc = combos_acc_sorted
       
        combos_bal_acc =  zip(feature_combos, bal_acc)
        combos_bal_acc_sorted = sorted(combos_bal_acc, key=lambda x: x[1], reverse=True)
        top_combos_bal_acc = combos_bal_acc_sorted
        print(f"Fold:{fold}")

        list_combos_acc.append(top_combos_acc)
        list_combos_bal_acc.append(top_combos_bal_acc)
        #last50_combos_acc = combos_acc_sorted[-50:]
        #last50_combos_bal_acc = combos_bal_acc_sorted[-50:]
    
    end = time.time()
    print(end-start)

    start = time.time()
    #Go through all the list of feature combos for the 5 CV folds and average the scores for all the feature combos
    combos_folds_acc = {}
    for fold in (list_combos_acc[0], list_combos_acc[1], list_combos_acc[2], list_combos_acc[3], list_combos_acc[4]):
        for c, a in fold:
            combos_folds_acc.setdefault(c, []).append(a)   

    acc_averages = [(c, sum(a)/len(a)) for c, a in combos_folds_acc.items() if len(a) > 4]
    acc_top_averages = sorted(acc_averages, key=lambda x: x[1], reverse=True)  

    combos_folds_bal_acc = {}
    for fold in (list_combos_bal_acc[0], list_combos_bal_acc[1], list_combos_bal_acc[2], list_combos_bal_acc[3], list_combos_bal_acc[4]):
        for c, ba in fold:
            combos_folds_bal_acc.setdefault(c, []).append(ba)

    ball_acc_averages = [(c, sum(ba)/len(ba)) for c, ba in combos_folds_bal_acc.items() if len(ba) > 4]
    ball_acc_top_averages = sorted(ball_acc_averages, key=lambda x: x[1], reverse=True)  
    
    end = time.time()
    print(end-start)
   
    print("\n\n Top accuracy:\n", acc_top_averages[:3])
    print("\n Number of matched combos between folds in accuracy:\n", len(acc_top_averages))
    print("\n\n Top balanced accuracy:\n", ball_acc_top_averages[:3])
    print("\n Number of matched combos between folds balanced accuracy:\n", len(ball_acc_top_averages))

