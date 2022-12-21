import pandas as pd
import numpy as np
import os
from sklearn.utils import shuffle

import torch
from torch.nn import (Sequential, Conv1d, Linear, MaxPool2d, ReLU, 
                    AvgPool2d, MaxPool1d, CrossEntropyLoss, Module, Flatten)
from torchinfo import summary
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import optuna
from sklearn.model_selection import train_test_split
import torch.optim as optim
import json
import time 


import sys

device = "cuda"
ids = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10", "chb11",
            "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20", "chb21",
            "chb22", "chb23", "chb24"]
path = "Data/Windows/"
 

def get_dataloader(X, Y):

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, 16384, shuffle=True) # create your dataloader 2048

    return my_dataloader  

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

def get_data_kfold(path, data_balanced = True):

    x_data = np.empty((0,21,128))
    y_data = np.empty((0))
    e = 1

    files = os.listdir(path)

    for file in files:

        if e % 15 == 0:
            print("Data Loading Progress:", round(e  / len(files) * 100, 2), "%")

        if ".pkl" not in file:
            continue

        df = pd.read_pickle(path + file)
        df = df[(df["Type"] != 2) & (df["Type"] != 3)]

        df_x = df.iloc[:, 3:].to_numpy()
        df_y = df.iloc[:, 1].to_numpy()

        if len(df_y) == 0:
            continue

        y_data = np.append(y_data, np.array(df_y), axis = 0)

        sub_train_x = []
        for i in range(len(df_x)):
            row = np.array(list(df_x[i]), dtype=float)
            sub_train_x.append(row)

        x_data = np.append(x_data, sub_train_x, axis = 0)

        e += 1
    
    print("x_data shape", x_data.shape)
    return x_data, y_data

def get_data_loo(path, data_balanced = True, step = 1):
    # Returns dict with window data for every patient 

    x_data = dict.fromkeys(ids, np.empty((0,21,128)))
    y_data = dict.fromkeys(ids, np.empty((0)))
    total, e = 0, 0
    files = os.listdir(path)

    for file in os.listdir(path):
        #print(file)
        if ".pkl" not in file:
            continue
        
        if e % 15 == 0:
            print("Data Loading Progress:", round(e  / len(files) * 100, 2), "%")

        patID = file.split("_")[0][:5]        
 
        df = pd.read_pickle(path + file)
        df = df[(df["Type"] != 2) & (df["Type"] != 3)]

        df_x = df.iloc[:, 3:].to_numpy()
        df_y = df.iloc[:, 1].to_numpy()

        if len(df_y) == 0:
            continue

        
        df_x, df_y = df_x[::step], df_y[::step]

        y_data[patID] = np.append(y_data[patID], np.array(df_y), axis = 0)

        sub_train_x = []
        for i in range(len(df_x)):
            row = np.array(list(df_x[i]), dtype=float)
            sub_train_x.append(row)
        total += len(df_y)
        x_data[patID] = np.append(x_data[patID], sub_train_x, axis = 0)
        e+= 1


    return x_data, y_data

class EpilepsyClassifier(Module):

    def __init__(self):
        
        super().__init__()

        self.n_channels = 21
        self.length = 128
        self.kernel_size = 4
        self.kernel_size_max_pool = 2

        self.model = Sequential(

            # Data Fusion Unit
            AvgPool2d((self.n_channels, 1)),

            # Network Unit
            Conv1d(in_channels=1, out_channels=16, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.kernel_size_max_pool),
    
            Conv1d(in_channels=16, out_channels=32, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.kernel_size_max_pool),

            Conv1d(in_channels=32, out_channels=64, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.kernel_size_max_pool),

            Flatten(),

            Linear(1024, 128),
            Linear(128, 2),
            
        )

    def forward(self, x):
        return self.model(x)


    def show_summary(self):
        summary((self.model), (32, 21, 128))

class EpilepsyClassifier_custom(Module):

    def __init__(self, parameters):
        
        super().__init__()

        self.n_channels = 21
        self.kernel_size_max_pool = parameters["kernel_size"]
        self.stride = self.kernel_size_max_pool

        self.model = Sequential(

            # Data Fusion Unit
            AvgPool2d((self.n_channels, 1)),
            # Network Unit
            Conv1d(in_channels=1, out_channels=16, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.stride),
            Conv1d(in_channels=16, out_channels=32, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.stride),
            Conv1d(in_channels=32, out_channels=64, kernel_size = 1 ,stride=1,), #padding=1, padding_mode="zeros"), 
            ReLU(),
            MaxPool1d(self.kernel_size_max_pool, stride=self.stride),
            Flatten(),
            Linear(64*int(128 / (self.kernel_size_max_pool ** 3)), 64),
            Linear(64, 2),
            
        )


    def forward(self, x):
        return self.model(x)


    def show_summary(self):
        print(summary((self.model), (32, 21, 128)))

def train_model(epoch, criterion, model, optimizer, dataloader):
    # Loop to train model

    total_loss = 0

    # Put model in train mode
    model.train()

    for batch_idx, (data, target) in enumerate(dataloader):

        optimizer.zero_grad()

        # Send data to GPU
        target = target.type(torch.LongTensor) 
        data, target = data.to(device), target.to(device)

        # Prediction
        output = model(data)

        # Loss computation & learning
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()


        if (batch_idx+1) % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), loss.item()))

        total_loss += loss.item()

    total_loss /= len(dataloader.dataset)

    return total_loss

@torch.no_grad()
def validate(criterion, model, dataloader, metrics_data = None, fold = None, optuna_mode = False):
    # Validate model

    val_loss = 0
    f1_loss_0, f1_loss_1 = 0, 0
    precision_0, precision_1 = 0, 0
    recall_0, recall_1 = 0, 0
    accuracy = 0

    correct = 0

    model.eval()
    num_tests = 0
    for data, target in dataloader:

        # Send data to GPU
        target = target.type(torch.LongTensor) 
        data, target = data.to(device), target.to(device)

        # Prediction
        output = model(data)

        # Loss computation & learning
        loss = criterion(output, target)
        val_loss += loss.item()

        # Get the index of max log probability
        pred = output.data.max(1, keepdim = True)[1]

        # Sum correct predictions
        correct += pred.eq(target.view_as(pred)).sum().item()

        target = target.cpu()
        pred = pred.cpu()

        f1_loss = f1_score(pred, target, average=None)
        f1_loss_0 += f1_loss[0]
        f1_loss_1 += f1_loss[1]

        if not optuna_mode:

            # Metrics
            accuracy += accuracy_score(target, pred)

            precision = list(precision_score(target, pred, average = None, zero_division=True))
            precision_0 += precision[0]
            precision_1 += precision[1]

            recall = list(recall_score(target, pred, average = None, zero_division=True))
            recall_0 += recall[0]
            recall_1 += recall[1]

            f1_loss = f1_score(pred.cpu(), target.cpu(), average=None)
            f1_loss_0 += f1_loss[0]
            f1_loss_1 += f1_loss[1]
            
        num_tests += 1

    f1_loss_0, f1_loss_1 = f1_loss_0 / num_tests, f1_loss_1 / num_tests

    if optuna_mode:
        return [f1_loss_0, f1_loss_1]

    else:
    
        accuracy /= num_tests
        precision_0, precision_1 = precision_0 / num_tests, precision_1 / num_tests    
        recall_0, recall_1 = recall_0 / num_tests, recall_1 / num_tests
        
        metrics_data[fold]["accuracy"] = accuracy
        metrics_data[fold]["precision"] = [precision_0, precision_1]
        metrics_data[fold]["recall"] = [recall_0, recall_1]
        metrics_data[fold]["f1-score"] = [f1_loss_0, f1_loss_1]

        return metrics_data


def k_fold(X, Y, n_splits = 20, n_epochs  = 20, balanced = True, parameters = None):
    
    initial_time = time.time()
    skf = StratifiedKFold(n_splits=n_splits)

    indexs = [(train_index, test_index) for train_index, test_index in skf.split(np.arange(len(Y)), Y)]

    #f1_score_splits = []
    metrics = ["accuracy", "precision", "recall", "f1-score"]
    metrics_data = dict.fromkeys([i for i in range(n_splits)])
    
    for i in range(n_splits):
        metrics_data[i] = dict.fromkeys(metrics, 0)
    
    fold = 0
    for train_index, test_index in indexs:
        print("Fold", fold)
        start_time = time.time()
        # Init model & parameters
        model = EpilepsyClassifier_custom({"kernel_size":parameters[0]}).to("cuda")
        opt = getattr(optim, parameters[2])(model.parameters(), lr=parameters[1])
        loss_fn = CrossEntropyLoss()
        epoch = 0

        # Train / Test dataset
        X_train, X_test = X[train_index, :], X[test_index, :]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if balanced:
            X_train, Y_train = balance_data(X_train, Y_train)

        # Dataloaders
        train_dataloader = get_dataloader(X_train, Y_train)
        test_dataloader = get_dataloader(X_test, Y_test)

        for epoch in range(n_epochs):
            # Model train
            train_loss = train_model(epoch=epoch, criterion=loss_fn, model=model, optimizer=opt, dataloader=train_dataloader)
            #losses["train"].append(train_loss)


        # Model test
        validate(criterion=loss_fn, model=model, dataloader=test_dataloader, metrics_data = metrics_data, fold=fold)
        metrics_data[fold]["time"] = str(round(time.time() - start_time,3))
        fold += 1

    metrics_data["Time"] = str(round(time.time() - initial_time,3))
    metrics_data["Folds"] = n_splits
    metrics_data["Step"] = 0
    metrics_data["Balanced"] = balanced
    metrics_data["kernel_size"] = parameters[0]
    metrics_data["lr"] = parameters[1]
    metrics_data["optimizer"] = parameters[2]

    if balanced:
      # Save metric data as json
      with open("Results/CNN_KFold_metrics_balanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
          
    else:
      # Save metric data as json
      with open("Results/CNN_KFold_metrics_unbalanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)

    return metrics_data

def format_data_from_loo(X_dict, Y_dict, patID):
    # Returns data from the patID as test and the rest as train
    
    X_test, Y_test = X_dict[patID], Y_dict[patID]

    X_train = np.empty((0,21,128))
    Y_train = np.empty((0))

    for patient in Y_dict:
        if patient != patID:
            X_train = np.append(X_train, X_dict[patient], axis = 0)
            Y_train = np.append(Y_train, Y_dict[patient], axis = 0)

    return X_train, X_test, Y_train, Y_test
        

def leave_one_out(X_dict, Y_dict, n_epochs=20, balanced = True, parameters = None, step = 5):

    initial_time = time.time()

    metrics = ["accuracy", "precision", "recall", "f1-score"]
    metrics_data = dict.fromkeys([pat for pat in Y_dict])
        

    # Iteration over all patients
    for patID in Y_dict:

        metrics_data[patID] = dict.fromkeys(metrics, 0)

        print("PatID", patID, "Balanced:", balanced)
        # Init model
        start_time = time.time()

        model = EpilepsyClassifier_custom({"kernel_size":parameters[0]}).to("cuda")
        opt = getattr(optim, parameters[2])(model.parameters(), lr=parameters[1])
        loss_fn = CrossEntropyLoss()

        # Get data
        X_train, X_test, Y_train, Y_test = format_data_from_loo(X_dict, Y_dict, patID) 
        
        if balanced:
            X_train, Y_train = balance_data(X_train, Y_train)

        # Dataloader
        train_dataloader = get_dataloader(X_train, Y_train)
        test_dataloader = get_dataloader(X_test, Y_test)

        for epoch in range(n_epochs):
            # Model train
            train_loss = train_model(epoch=epoch, criterion=loss_fn, model=model, optimizer=opt, dataloader=train_dataloader)

        # Model test
        validate(criterion=loss_fn, model=model, dataloader=test_dataloader, metrics_data=metrics_data, fold=patID)
        metrics_data[patID]["time"] = str(round(time.time() - start_time,3))

    metrics_data["Time"] = str(round(time.time() - initial_time,3))
    metrics_data["Folds"] = len(Y_dict)
    metrics_data["Step"] = step # modificar
    metrics_data["Balanced"] = balanced
    metrics_data["kernel_size"] = parameters[0]
    metrics_data["lr"] = parameters[1]
    metrics_data["optimizer"] = parameters[2]
    
    if balanced:
      # Save metric data as json
      with open("Results/CNN_LOO_metrics_balanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
          
    else:
      # Save metric data as json
      with open("Results/CNN_LOO_metrics_unbalanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
     

    return

def optuna_study(X, Y, balance = True):

    direction_list = ["maximize" for i in range(2)]
    study = optuna.create_study(directions=direction_list)#, pruner=optuna.pruners.MedianPruner())

    study.optimize(lambda trial: objective(trial, X, Y, balanced=balance), n_trials = 30)

    print('Number of finished trials:', len(study.trials))

    fig = optuna.visualization.plot_pareto_front(study, target_names = ["F1_Score_0", "F1_Score_1"])
    df = study.trials_dataframe(attrs=["number", "value", "duration", "params", "state"])

    if balance:

        df.to_pickle("Results/CNN_Balanced_Study.pkl")
        fig.write_html("Results/CNN_Balanced_plotly.html")

    else:
        df.to_pickle("Results/CNN_Unbalanced_Study.pkl")
        fig.write_html("Results/CNN_Unbalanced_plotly.html")

    return study

def objective(trial, X, Y, balanced = True):
    print("Trial number", trial.number, "of 30")
    skf = StratifiedKFold(n_splits=5)

    indexs = [(train_index, test_index) for train_index, test_index in skf.split(np.arange(len(Y)), Y)]

    f1_score_splits = []

    param = {"kernel_size": trial.suggest_int("kernel_size", 2,4),
                "lr": trial.suggest_float("lr", 1e-5, 1e-1),
                "optimizer": trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]), 
                }

    i = 1
    for train_index, test_index in indexs:
        if (i%5) == 0:
            print("Split number", i, "of 5")
        # Init model & parameters
        model = EpilepsyClassifier_custom({"kernel_size":param["kernel_size"]}).to("cuda")
        opt = getattr(optim, param['optimizer'])(model.parameters(), lr=param["lr"])
        loss_fn = CrossEntropyLoss()
        epoch = 0

        # Train / Test dataset
        X_train, X_test = X[train_index, :], X[test_index, :]
        Y_train, Y_test = Y[train_index], Y[test_index]

        if balanced:
            X_train, Y_train = balance_data(X_train, Y_train)

        # Dataloaders
        train_dataloader = get_dataloader(X_train, Y_train)
        test_dataloader = get_dataloader(X_test, Y_test)

        for epoch in range(20):
            # Model train
            train_model(epoch=epoch, criterion=loss_fn, model=model, optimizer=opt, dataloader=train_dataloader)


        # Model test
        f1_score_validation = validate(criterion=loss_fn, model=model, dataloader=test_dataloader, optuna_mode=True)
        f1_score_splits.append(f1_score_validation)
        i += 1

    f1_score_0 = sum([x[0] for x in f1_score_splits]) / len(f1_score_splits)
    f1_score_1 = sum([x[1] for x in f1_score_splits]) / len(f1_score_splits)

    return f1_score_0, f1_score_1



if __name__ == "__main__":

    # Optuna study KFOLD
    
    X, Y = get_data_kfold(path, data_balanced=True)
    study = optuna_study(X, Y, True)
    '''
    X, Y = get_data_kfold(path, data_balanced=False)
    study = optuna_study(X, Y, False)
    '''

    '''
    # Kfold with optimized parameters
    params = [4, 0.04578, "SGD"]
    X, Y = get_data_kfold(path, data_balanced=True)
    k_fold(X, Y, n_splits=20, n_epochs=30, balanced=True, parameters=params)
    
    params = [3, 0.01489, "Adam"]
    X, Y = get_data_kfold(path, data_balanced=False)
    k_fold(X, Y, n_splits=20, n_epochs=30, balanced=False, parameters=params)
    '''

    '''
    # Leave One Out
    step = 1
    X_dict, Y_dict = get_data_loo(path=path, data_balanced=True, step = step)
    params = [4, 0.04578, "SGD"]
    leave_one_out(X_dict, Y_dict, n_epochs=25, balanced=True, parameters=params, step = step)

    X_dict, Y_dict = get_data_loo(path=path, data_balanced=False, step = step)
    params = [3, 0.01489, "Adam"]
    leave_one_out(X_dict, Y_dict, n_epochs=25, balanced=False, parameters=params, step = step)
    '''
    
