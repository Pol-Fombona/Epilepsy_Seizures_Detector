import torch
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.nn import (Sequential, Conv1d, Linear, ReLU, Dropout,
                    AvgPool2d, MaxPool1d, CrossEntropyLoss, Module, Flatten)
import pandas as pd
import numpy as np
import os
import json
import time
import optuna
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch import nn as nn
import sys
from torchinfo import summary


device = "cuda"
ids = ["chb01", "chb02", "chb03", "chb04", "chb05", "chb06", "chb07", "chb08", "chb09", "chb10", "chb11",
            "chb12", "chb13", "chb14", "chb15", "chb16", "chb17", "chb18", "chb19", "chb20", "chb21",
            "chb22", "chb23", "chb24"]
path = "Data/Windows/"

def create_windows_for_transformer(path):

    model = EpilepsyClassifier().to(device)
    model.eval()

    for idx, file in enumerate(os.listdir(path)):
        print("Processing file", file)
        new_data = []

        x_data = np.empty((0,21,128))
        y_data = np.empty((0))
        
        df = pd.read_pickle(path+file)
        patID = file.split("_")[0][:5]

        df_x = df.iloc[:, 3:].to_numpy()
        df_y = df.iloc[:, 1].to_numpy()

        y_data = np.append(y_data, np.array(df_y), axis = 0)

        sub_train_x = []
        for i in range(len(df_x)):
            row = np.array(list(df_x[i]), dtype=float)
            sub_train_x.append(row)

        x_data = np.append(x_data, sub_train_x, axis = 0)
        
        tensor_x = torch.Tensor(x_data)
        tensor_y = torch.Tensor(y_data)

        dataloader = DataLoader(TensorDataset(tensor_x,tensor_y), 1, shuffle=False) # create your dataloader 2048

        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            output = model(data)
            output = output.cpu().detach().numpy()[0]
            new_data.append([y_data[batch_idx], output])
        
        df_output = pd.DataFrame(new_data, columns=["Type","Signal"])
        df_output.to_pickle("Data/Windows_Transformer/"+file)


def get_dataloader(X, Y):

    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(Y)

    my_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    my_dataloader = DataLoader(my_dataset, 256, shuffle=True) # create your dataloader 2048
    del my_dataset

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


def get_data(path, balanced = False, step = 1, number_of_windows_in_array = 4):

    print("Loading data...")

    x_data = dict.fromkeys(ids, np.empty((0, number_of_windows_in_array, 160)))
    y_data = dict.fromkeys(ids, np.empty((0)))

    for idx, file in enumerate(os.listdir(path)):
        patID = file.split("_")[0][:5]        
        df = pd.read_pickle(path + file)
        
        df = df[(df["Type"] != 2) & (df["Type"] != 3)] 
        df_x = df.iloc[:, 1].to_numpy()
        df_y = df.iloc[:, 0].to_numpy()

        if balanced:
            df_x, df_y = balance_data(df_x, df_y)

        if len(df_y) == 0:
            continue

        df_x, df_y = df_x[::step], df_y[::step]

        sub_data_x = []
        sub_data_y = []
        for i in range(0, len(df_x), number_of_windows_in_array):
            
            if (i + number_of_windows_in_array - 1) >= len(df_x):
                continue

            windows_in_array = []
            for e in range(number_of_windows_in_array):
                windows_in_array.append(np.array(list(df_x[i+e]), dtype=float))

            sub_data_x.append(windows_in_array)
            sub_data_y.append(df_y[i+number_of_windows_in_array-1])

        y_data[patID] = np.append(y_data[patID], sub_data_y, axis=0)
        x_data[patID] = np.append(x_data[patID], sub_data_x, axis=0)

    print("Loading finished")
    return x_data, y_data

def get_data_loo(path, data_balanced = True, step = 5):
    # Returns dict with window data for every patient 

    x_data = dict.fromkeys(ids, np.empty((0,21,128)))
    y_data = dict.fromkeys(ids, np.empty((0)))
    total, idx = 0, 0
    files = os.listdir(path)

    for idx, file in enumerate(os.listdir(path)):
        if ".pkl" not in file:
            continue
        
        if idx % 15 == 0:
            print("Data Loading Progress:", round(idx  / len(files) * 100, 2), "%")

        patID = file.split("_")[0][:5]        
 
        df = pd.read_pickle(path + file)
        df = df[(df["Type"] != 2) & (df["Type"] != 3)]

        df_x = df.iloc[:, 3:].to_numpy()
        df_y = df.iloc[:, 1].to_numpy()

        if data_balanced:
            df_x, df_y = balance_data(df_x, df_y)

        if len(df_y) == 0:
            continue

        
        #df_x, df_y = df_x[::step], df_y[::step]

        y_data[patID] = np.append(y_data[patID], np.array(df_y), axis = 0)

        sub_train_x = []
        for i in range(len(df_x)):
            row = np.array(list(df_x[i]), dtype=float)
            sub_train_x.append(row)

        total += len(df_y)
        x_data[patID] = np.append(x_data[patID], sub_train_x, axis = 0)

    print("Total size", total)

    return x_data, y_data


def format_data_from_loo(X_dict, Y_dict, patID, number_of_windows_in_array):
    # Returns data from the patID as test and the rest as train
    
    X_test, Y_test = X_dict[patID], Y_dict[patID]

    X_train = np.empty((0, number_of_windows_in_array,160))
    Y_train = np.empty((0))

    for patient in Y_dict:
        if patient != patID:
            X_train = np.append(X_train, X_dict[patient], axis = 0)
            Y_train = np.append(Y_train, Y_dict[patient], axis = 0)

    return X_train, X_test, Y_train, Y_test

class EpilepsyClassifier(Module):

    def __init__(self):
        
        super().__init__()

        self.n_channels = 21
        self.length = 128
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

            # Linear layers
            Linear(1024, 192),
            ReLU(),
            Dropout(0.1),
            Linear(192, 128),
            ReLU(),
            Dropout(0.1),
            Linear(128, 160),
            ReLU(),
            Dropout(0.5),
            Linear(160, 2)
        )

        self.init_weights("EpilepsyClassifier_model_kfold.pt")


    def forward(self, x):
        return self.model(x)
    

    def init_weights(self, path):
        # Init weigths from pretrained model
        self.model = torch.nn.Sequential(*(list(self.model.children())[:-1]))
        self.model.load_state_dict(torch.load(path), strict=False)









class TransformerModel(nn.Module):
    """Container module with an encoder, a recurrent or transformer module, and a decoder."""
    
    def __init__(self, inputmodule_params,net_params,outmodule_params, n_windows):
        super().__init__()

        
        self.model_type = 'Transformer'
        ### Input Parameters
        self.ninp = inputmodule_params['emsize']
        self.nhead=net_params['nhead']
        self.nhid=net_params['nhid']
        self.dropout=net_params['dropout']
        if net_params['dropout'] is None:
            self.dropout=0.5
        self.nlayers=net_params['nlayers']
        self.activation=outmodule_params['activation']

        ### Architecture
        encoder_layers = TransformerEncoderLayer(self.ninp, self.nhead, self.nhid, self.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.nlayers)
        self.linear = nn.Linear(n_windows*self.ninp, 2)
        self.flatten = nn.Flatten()

    def forward(self, src, src_mask=None):

        #src = self.encoder(src)

        output = self.transformer_encoder(src,src_mask) # Dim (batch, num_windows = 12, output_classifier = 160)
        output = self.flatten(output)
        output = self.linear(output) # Dim (batch, num_windows=12, num_classes = 2)

        return output


    def show_summary(self):
        summary((self.model), (256, 4, 160))

def train_model(epoch, criterion, model, optimizer, dataloader):
    # Loop to train model

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

        del data, target, output, loss

    return 

@torch.no_grad()
def validate(criterion, model, dataloader, metrics_data = None, fold = None, optuna_mode = False):
    # Validate model

    f1_loss_0, f1_loss_1 = 0, 0
    precision_0, precision_1 = 0, 0
    recall_0, recall_1 = 0, 0
    accuracy = 0

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
        # Get the index of max log probability
        pred = output.data.max(1, keepdim = True)[1]
        #print(pred)
        target = target.cpu()
        pred = pred.cpu()

        f1_loss = f1_score(target, pred, average=None)

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


def leave_one_out(X_dict, Y_dict, n_epochs=20, balanced = True, params = None, step = 0, number_of_windows_in_array=4):

    initial_time = time.time()

    metrics = ["accuracy", "precision", "recall", "f1-score"]
    metrics_data = dict.fromkeys([pat for pat in Y_dict])
        
    inputmodule_params = {"emsize":160}

    net_params = {"nhead":params["nhead"], "nhid":params["nhid"], 
                    "dropout": params["dropout"], "nlayers":params["nlayers"]}

    outmodule_params = {"activation":False}

    # Iteration over all patients
    for patID in Y_dict:

        metrics_data[patID] = dict.fromkeys(metrics, 0)

        print("PatID", patID, "Balanced:", balanced)
        # Init model
        start_time = time.time()

        model = TransformerModel(inputmodule_params,net_params,outmodule_params, n_windows=number_of_windows_in_array).to(device) 
        opt = getattr(optim, params['optimizer'])(model.parameters(), lr=params["lr"])
        loss_fn = CrossEntropyLoss()

        # Get data
        X_train, X_test, Y_train, Y_test = format_data_from_loo(X_dict, Y_dict, patID, number_of_windows_in_array) 
        
        X_train, Y_train = balance_data(X_train, Y_train)

        if balanced:
            X_test, Y_test = balance_data(X_test, Y_test)


        # Dataloader
        train_dataloader = get_dataloader(X_train, Y_train)
        test_dataloader = get_dataloader(X_test, Y_test)

        for epoch in range(n_epochs):
            if (epoch % 5) == 0:
                print("Epoch:", epoch)
            # Model train
            train_model(epoch=epoch, criterion=loss_fn, model=model, optimizer=opt, dataloader=train_dataloader)

        # Model test
        validate(criterion=loss_fn, model=model, dataloader=test_dataloader, metrics_data=metrics_data, fold=patID, optuna_mode=False)
        metrics_data[patID]["time"] = str(round(time.time() - start_time,3))

    metrics_data["Time"] = str(round(time.time() - initial_time,3))
    metrics_data["Folds"] = len(Y_dict)
    metrics_data["Step"] = step # modificar
    metrics_data["Balanced"] = balanced
    metrics_data["lr"] = params["lr"]
    metrics_data["optimizer"] = params["optimizer"]
    
    if balanced:
      # Save metric data as json
      with open("Results_Pol/Transfomer_LOO_metrics_balanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
          
    else:
      # Save metric data as json
      with open("Results_Pol/Transformer_LOO_metrics_unbalanced.json", "w") as outfile:
          json.dump(metrics_data, outfile)
     

    return

def optuna_study(dict_X, dict_Y, balance = True, n_epochs = 20, n_trials = 30, number_of_windows_in_array = 4):

    direction_list = ["maximize" for i in range(2)]
    study = optuna.create_study(directions=direction_list)#, pruner=optuna.pruners.MedianPruner())

    study.optimize(lambda trial: objective(trial, dict_X, dict_Y, balanced=balance, n_epochs=n_epochs, number_of_windows_in_array=number_of_windows_in_array), n_trials = n_trials)

    print('Number of finished trials:', len(study.trials))

    fig = optuna.visualization.plot_pareto_front(study, target_names = ["F1_Score_0", "F1_Score_1"])
    df = study.trials_dataframe(attrs=["number", "value", "duration", "params", "state"])

    if balance:

        df.to_pickle("Results/Transfomer_LOO_Balanced_Study.pkl")
        fig.write_html("Results/Transfomer_LOO_Balanced_plotly.html")

    else:
        df.to_pickle("Results/Transfomer_LOO_Unbalanced_Study.pkl")
        fig.write_html("Results/Transfomer_LOO_Unbalanced_plotly.html")

    return study

def objective(trial, X_dict, Y_dict, n_epochs=20, balanced = True, number_of_windows_in_array=4):
    print("Trial number", trial.number)

    initial_time = time.time()

    f1_score_splits = []

    params = {"nhead":trial.suggest_categorical("nhead", [2,4,8,10,16]), 
                "nhid":trial.suggest_int("nhid", 100, 1000, step=50), 
                "nlayers":trial.suggest_int("nlayers", 2, 10),
                "dropout":trial.suggest_float("dropout", 0.2, 0.8),
                "optimizer":trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"]),
                "lr":trial.suggest_float("lr", 1e-5, 1e-1)}

    inputmodule_params = {"emsize":160}

    net_params = {"nhead":params["nhead"], "nhid":params["nhid"], 
                    "dropout": params["dropout"], "nlayers":params["nlayers"]}

    outmodule_params = {"activation":True}


    for patID in Y_dict:

        print("PatID", patID, "Balanced:", balanced)

        model = TransformerModel(inputmodule_params,net_params,outmodule_params, n_windows=number_of_windows_in_array).to(device)
        opt = getattr(optim, params['optimizer'])(model.parameters(), lr=params["lr"])
        loss_fn = CrossEntropyLoss()

        X_train, X_test, Y_train, Y_test = format_data_from_loo(X_dict, Y_dict, patID, number_of_windows_in_array) 

        if balanced:
            X_train, Y_train = balance_data(X_train, Y_train)

        # Dataloader
        train_dataloader = get_dataloader(X_train, Y_train)
        test_dataloader = get_dataloader(X_test, Y_test)

        for epoch in range(n_epochs):
            if (epoch % 5) == 0:
                print("Epoch:", epoch)

            # Model train
            train_loss = train_model(epoch=epoch, criterion=loss_fn, model=model, optimizer=opt, dataloader=train_dataloader)

        # Model test
        f1_score_validation = validate(criterion=loss_fn, model=model, dataloader=test_dataloader, optuna_mode=True)
        f1_score_splits.append(f1_score_validation)
    
    f1_score_0 = sum([x[0] for x in f1_score_splits]) / len(f1_score_splits)
    f1_score_1 = sum([x[1] for x in f1_score_splits]) / len(f1_score_splits)

    print("Time:", str(round(time.time() - initial_time,2)))
    return f1_score_0, f1_score_1


if __name__ == "__main__":
    
    # Optuna study
    number_of_windows_in_array = 12
    #X_dict, Y_dict = get_data("Data/Windows_Transformer/", balanced=False, step=1, number_of_windows_in_array=number_of_windows_in_array)
    #optuna_study(X_dict, Y_dict, balance=True, n_epochs = 20, n_trials = 20, number_of_windows_in_array=number_of_windows_in_array)

    # Best params executions
    X_dict, Y_dict = get_data("Data/Windows_Transformer/", balanced=False, step=1, number_of_windows_in_array=number_of_windows_in_array)

    params = {"nhead":4, "nhid":350,"nlayers":2, "dropout":0.68846340, "optimizer":"SGD", "lr":0.07107175}
    # Test & Train Balanced
    leave_one_out(X_dict, Y_dict, n_epochs=50, balanced = True, params=params, number_of_windows_in_array=number_of_windows_in_array)
    
    # Test unbalanced
    leave_one_out(X_dict, Y_dict, n_epochs=50, balanced = False, params=params, number_of_windows_in_array=number_of_windows_in_array)
