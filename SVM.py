#!/usr/bin/env python

import pandas as pd
import numpy as np
import os
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import time
from threading import Thread
import json
from sklearn.model_selection import GridSearchCV
import sys
import optuna
from sklearn.model_selection import train_test_split
from sklearnex import patch_sklearn 
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.utils import shuffle

def get_svm_leave_one_out(EEG_Feat, EEG_GT, GROUP_METADATA, n_splits = 10, step = 5, balance_train_data = False, parameters=None): # provar amb 5
    # Threading
    
    temp_time = time.time()
    skf = LeaveOneGroupOut()
    skf.get_n_splits(groups=GROUP_METADATA)
    metrics = ["accuracy", "precision", "recall", "f1-score"]

    
    # Generate metric dictionary
    metrics_data = dict.fromkeys([i for i in range(n_splits)])
    for i in range(n_splits):
        metrics_data[i] = dict.fromkeys(metrics, 0)

    temp = [(train_index, test_index) for train_index, test_index in skf.split(np.arange(len(EEG_GT)), EEG_GT, GROUP_METADATA)]

    # Loop over folds (mod 5)
    for i in range(0, n_splits, 5):
        t1 = Thread(target=svm_loop, args=(temp[i][0], temp[i][1], step, i, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t2 = Thread(target=svm_loop, args=(temp[i+1][0], temp[i+1][1], step, i+1, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t3 = Thread(target=svm_loop, args=(temp[i+2][0], temp[i+2][1], step, i+2, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t4 = Thread(target=svm_loop, args=(temp[i+3][0], temp[i+3][1], step, i+3, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t5 = Thread(target=svm_loop, args=(temp[i+4][0], temp[i+4][1], step, i+4, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        threads = [t1, t2, t3, t4, t5]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    end_time = round(time.time() - temp_time, 3)
    
    # Average data
    #metrics_data = adjust_metrics_data(metrics_data, n_splits, metrics)
    
    # Add extra metric data
    metrics_data["Time"] = end_time
    metrics_data["Folds"] = n_splits
    metrics_data["Step"] = step
    metrics_data["Balanced"] = balance_train_data
    metrics_data["C"] = parameters[0]
    metrics_data["Kernel"] = parameters[1]
    metrics_data["Gamma"] = parameters[2]
    
    if balance_train_data:
      # Save metric data as json
      with open("Results/Leave_one_out_metrics_balanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
          
    else:
      # Save metric data as json
      with open("Results/Leave_one_out_metrics_unbalanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
        
    #print("Elapsed time", str(end_time))
    
    return metrics_data


def get_svm(EEG_Feat, EEG_GT, n_splits = 10, step = 5, balance_train_data = False, parameters=None): # provar amb 5
    # Threading
    
    temp_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits)
    metrics = ["accuracy", "precision", "recall", "f1-score"]

    
    # Generate metric dictionary
    metrics_data = dict.fromkeys([i for i in range(n_splits)])
    for i in range(n_splits):
        metrics_data[i] = dict.fromkeys(metrics, 0)
    temp = [(train_index, test_index) for train_index, test_index in skf.split(np.arange(len(EEG_GT)), EEG_GT)]
    
    # Loop over folds (mod 5)
    for i in range(0, n_splits, 5):
        t1 = Thread(target=svm_loop, args=(temp[i][0], temp[i][1], step, i, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t2 = Thread(target=svm_loop, args=(temp[i+1][0], temp[i+1][1], step, i+1, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t3 = Thread(target=svm_loop, args=(temp[i+2][0], temp[i+2][1], step, i+2, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t4 = Thread(target=svm_loop, args=(temp[i+3][0], temp[i+3][1], step, i+3, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        t5 = Thread(target=svm_loop, args=(temp[i+4][0], temp[i+4][1], step, i+4, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters))
        threads = [t1, t2, t3, t4, t5]
        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    end_time = round(time.time() - temp_time, 3)
        
    # Add extra metric data
    metrics_data["Time"] = end_time
    metrics_data["Folds"] = n_splits
    metrics_data["Step"] = step
    metrics_data["Balanced"] = balance_train_data
    metrics_data["C"] = parameters[0]
    metrics_data["Kernel"] = parameters[1]
    metrics_data["Gamma"] = parameters[2]
    
    if balance_train_data:
      # Save metric data as json
      with open("Results/StratifiedKFold_metrics_balanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
          
    else:
      # Save metric data as json
      with open("Results/StratifiedKFold_metrics_unbalanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
        
    #print("Elapsed time", str(end_time))
    
    return metrics_data
    

def get_svm_lineal(EEG_Feat, EEG_GT, n_splits = 10, step = 5):
    # Lineal svm

    temp_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits)
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    
    metrics_data = dict.fromkeys([i for i in range(n_splits)])
    
    for i in range(n_splits):
        metrics_data[i] = dict.fromkeys(metrics, 0)
        
    temp = [(train_index, test_index) for train_index, test_index in skf.split(np.arange(len(EEG_GT)), EEG_GT)]
    
    for i in range(0, n_splits):
        svm_loop(temp[i][0], temp[i][1], step, i, metrics_data, EEG_Feat, EEG_GT)
        
        
    end_time = round(time.time() - temp_time, 3)
    
    metrics_data = adjust_metrics_data(metrics_data, n_splits, metrics)
    metrics_data["Time"] = end_time
    metrics_data["Folds"] = n_splits
    metrics_data["Step"] = step
    
    with open("Results/metrics_average.json", "w") as outfile:
        json.dump(metrics_data, outfile)
    #print("Elapsed time", str(end_time))
    
    return metrics_data
    
def svm_loop(train_index, test_index, step, fold, metrics_data, EEG_Feat, EEG_GT, balance_train_data, parameters):
        
        print("Fold", fold, "starting")
        
        start_time = time.time()
        # Train model
        SVM = SVC(C=parameters[0],kernel=parameters[1], gamma=parameters[2])

        X_train, X_test = EEG_Feat[train_index[0::step],:], EEG_Feat[test_index[0::step],:]
        Y_train, Y_test = EEG_GT[train_index[0::step]], EEG_GT[test_index[0::step]]

        if balance_train_data:
            X_train, Y_train = balance_data(X_train, Y_train)

        X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))


        SVM.fit(X_train, Y_train)
        prediction = SVM.predict(X_test)

        end_time = time.time()

        metrics_data[fold]["accuracy"] = accuracy_score(Y_test, prediction)
        metrics_data[fold]["precision"] = list(precision_score(Y_test, prediction, average = None, zero_division=True))
        metrics_data[fold]["recall"] = list(recall_score(Y_test, prediction, average = None, zero_division=True))
        metrics_data[fold]["f1-score"] = list(f1_score(Y_test, prediction, average = None))
        metrics_data[fold]["time"] = str(round(end_time - start_time,3))

        print("Fold", fold, "- Time Elapsed", str(round(end_time - start_time,3)))

        return


def objective(trial, EEG_Feat, EEG_GT, balance_train_data, step):
                
        X_train, X_test, Y_train, Y_test = train_test_split(EEG_Feat, EEG_GT, test_size = 0.15)

        if step > 1:
            X_train, Y_train = X_train[::step], Y_train[::step]
            X_test, Y_test = X_test[::step], Y_test[::step]

        if balance_train_data:
            X_train, Y_train = balance_data(X_train, Y_train)

        
        param = {"C": trial.suggest_float("C", 0.0001, 0.1),
                    "kernel": trial.suggest_categorical("kernel", ["rbf","sigmoid"]),#["rbf", "poly", "sigmoid"]),
                    "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
                }

        X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))

        SVM = SVC(**param)
        SVM.fit(X_train, Y_train)

        prediction = SVM.predict(X_test)

        #accuracy = accuracy_score(Y_test, prediction)
        #precision0, precision1 = list(precision_score(Y_test, prediction, average = None, zero_division=True))
        #recall0, recall1 = list(recall_score(Y_test, prediction, average = None, zero_division=True))
        f1score0, f1score1 = list(f1_score(Y_test, prediction, average = None))

        return f1score0, f1score1

def svm_loop_grid_search(train_index, test_index, step, fold, metrics_data, EEG_Feat, EEG_GT):
        
        #print("Fold", fold, "starting")
        parameters = {"C" : [0.001, 0.01], 
                      #"class_weight":["balanced", None],
                      "class_weight" : ["balanced"],
                      "kernel" : ["rbf"],
                      #"gamma" : [1, 0.1, 0.01, 0.001]
                      }
        
        start_time = time.time()
        # Train model
        SVM = SVC()
        grid = GridSearchCV(estimator=SVM, param_grid=parameters)
        X_train, X_test = EEG_Feat[train_index[0::step],:], EEG_Feat[test_index[0::step],:]
        X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
        X_test=np.reshape(X_test,(X_test.shape[0],X_test.shape[1]*X_test.shape[2]))
        Y_train, Y_test = EEG_GT[train_index[0::step]], EEG_GT[test_index[0::step]]

        grid.fit(X_train,Y_train)
        prediction = grid.predict(X_test)

        metrics_data[fold]["accuracy"] = accuracy_score(Y_test, prediction)
        metrics_data[fold]["precision"] = precision_score(Y_test, prediction, average = None, zero_division=True)
        metrics_data[fold]["recall"] = recall_score(Y_test, prediction, average = None, zero_division=True)
        metrics_data[fold]["f1-score"] = f1_score(Y_test, prediction, average = None)

        end_time = time.time()
        #print("Fold", fold, "- Time Elapsed", str(round(end_time - start_time,3)))
        return

def balance_data(x_data, y_data):
    # Balance class data (50/50)

    class_1_count = np.sum(y_data)

    # Get index of class 0 and shuffle
    class_0_index = np.where(y_data==0)[0]
    np.random.shuffle(class_0_index)
    class_0_index = list(class_0_index)

    # Keep index to remove data
    index_to_remove = class_0_index[:int(len(class_0_index)-class_1_count)]

    x_data = np.delete(x_data, index_to_remove, axis=0)
    y_data = np.delete(y_data, index_to_remove)   

    return x_data, y_data
        

def get_data(feature_path, metadata_path):
    # Combines all pateient data
    
    metadata_files = sorted(os.listdir(metadata_path))
    feature_files = sorted(os.listdir(feature_path))

    EEG_Feat, EEG_GT = np.zeros((0,21,9)), np.zeros(0)

    for meta_file, feat_file in zip(metadata_files, feature_files):
        
        sub_EEG_Feat = np.array(np.load(feature_path + feat_file)["EEG_Feat"])
        sub_EEG_GT = np.array(pd.read_parquet(metadata_path + meta_file, engine="fastparquet")["class"])
        EEG_Feat = np.concatenate((EEG_Feat, sub_EEG_Feat))
        EEG_GT = np.append(EEG_GT, sub_EEG_GT)
        
    EEG_Feat, EEG_GT = shuffle(EEG_Feat, EEG_GT)

    return EEG_Feat, EEG_GT


def get_data(feature_path, metadata_path):
    # Combines all patient data and return group
    
    metadata_files = sorted(os.listdir(metadata_path))
    feature_files = sorted(os.listdir(feature_path))

    EEG_Feat, EEG_GT, GROUP_METADATA = np.zeros((0,21,9)), np.zeros(0), np.zeros(0)

    for meta_file, feat_file in zip(metadata_files, feature_files):
        
        sub_EEG_Feat = np.array(np.load(feature_path + feat_file)["EEG_Feat"])
        sub_EEG_GT = np.array(pd.read_parquet(metadata_path + meta_file, engine="fastparquet")["class"])
        sub_GROUP = pd.read_parquet(metadata_path + meta_file, engine="fastparquet")["filename"]
        sub_GROUP = np.array(sub_GROUP.str.split("_").str[0])

        EEG_Feat = np.concatenate((EEG_Feat, sub_EEG_Feat))
        EEG_GT = np.append(EEG_GT, sub_EEG_GT)
        GROUP_METADATA = np.append(GROUP_METADATA, sub_GROUP)
        

    EEG_Feat, EEG_GT, GROUP_METADATA = shuffle(EEG_Feat, EEG_GT, GROUP_METADATA)

    return EEG_Feat, EEG_GT, GROUP_METADATA

    
def adjust_metrics_data(metrics_data, n_splits, metrics):
    # Average metric data
    
    metric_data_combined = dict.fromkeys(metrics, 0)
    
    for fold in metrics_data:
        for metric in metrics_data[fold]:
            metric_data_combined[metric] += metrics_data[fold][metric]
    
    metric_data_combined = {key: value / n_splits for key, value in metric_data_combined.items()}
    metric_data_combined["precision"] = list(metric_data_combined["precision"])
    metric_data_combined["recall"] = list(metric_data_combined["recall"])
    metric_data_combined["f1-score"] = list(metric_data_combined["f1-score"])
    
    return metric_data_combined
    

def optuna_study(EEG_Feat, EEG_GT, balance_train_data=False, step=5):

    direction_list = ["maximize" for i in range(2)]
    study = optuna.create_study(directions=direction_list)
    study.optimize(lambda trial: objective(trial, EEG_Feat, EEG_GT, balance_train_data, step), 
            n_trials=30)
            
    print('Number of finished trials:', len(study.trials))

    fig = optuna.visualization.plot_pareto_front(study, target_names = ["F1_Score_0", "F1_Score_1"])
    df = study.trials_dataframe(attrs=["number", "value", "duration", "params", "state"])

    if balance_train_data:

        df.to_pickle("Results/Balanced_Study.pkl")
        fig.write_html("Results/Balanced_plotly.html")

    else:
        df.to_pickle("Results/Unbalanced_Study.pkl")
        fig.write_html("Results/Unbalanced_plotly.html")

    return study


if __name__ == "__main__":
    
    patch_sklearn()
    
    feature_path = "Data/ClassicFeatures/"
    metadata_path = "Data/Metadata/"
    
    print("Obtaining data")
    EEG_Feat, EEG_GT, GROUP_METADATA = get_data(feature_path, metadata_path)

    '''
    print("Optuna Balanced")
    study = optuna_study(EEG_Feat, EEG_GT, balance_train_data=True, step=20)
    print("\nOptuna Unbalanced")
    study = optuna_study(EEG_Feat, EEG_GT, balance_train_data=False, step=20)
    '''
    # Obtinguts del study
    params_balanced = [0.076353, "rbf", "scale"]
    params_unbalanced = [0.052505, "rbf", "scale"]
    
    print("SVM in Progress")
    metrics_data = get_svm(EEG_Feat, EEG_GT, n_splits=30, step=5, balance_train_data = True, parameters = params_balanced)
    print("SVM Done")
    print("SVM Leave One Out Balanced in Progress")
    metrics_data = get_svm_leave_one_out(EEG_Feat, EEG_GT, GROUP_METADATA, n_splits=25, step=5, balance_train_data = True, parameters = params_balanced)
    print("SVM Leave One Out Balanced Done")

    print("SVM unbalanced starting")
    metrics_data = get_svm(EEG_Feat, EEG_GT, n_splits=30, step=5, balance_train_data = False, parameters = params_unbalanced)
    print("SVM Done") 
    print("SVM Leave One Out Unbalanced in Progress")
    metrics_data = get_svm_leave_one_out(EEG_Feat, EEG_GT, GROUP_METADATA, n_splits=25, step=5, balance_train_data = False, parameters = params_unbalanced)
    print("SVM Leave One Out Unbalanced Done")
    