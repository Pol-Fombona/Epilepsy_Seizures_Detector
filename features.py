import numpy as np
import sys
import os
import pandas as pd
from scipy.stats import kurtosis, skew, entropy


files = os.listdir("window_array")

last_patID = files[0].split("_")[0]
data = []
df_metadata = []
global_interval = {0:0, 1:0}
last_type = None
last_file = None


for file in files:

    patID = file.split("_")[0]

    df = pd.read_pickle("window_array\\" + file)
    df = df[(df["Type"]==1 ) | (df["Type"]==0)]

    if patID != last_patID:
        # Save data of last patID

        sampled = int(file.split("Sampled_")[1].split("-")[0])
        win_size = int(file.split("WinSize_")[1].split("-")[0])
        window = int(win_size/sampled)

        # Features
        df_to_save = pd.DataFrame(data, columns=list(df.columns[3:]))

        filename = last_patID+"_classicfeatures_seizure_EEGwindow_"+str(window)+".pkl"
        df_to_save.to_pickle("features\\"+filename)

        print(filename, "--- Saved Correctly")

        # Metadata
        metadata_filename = last_patID+"_seizure_metadata_"+str(window)+".pkl"

        df_metadata_to_save = pd.DataFrame(df_metadata, columns=["Class", "Filename_Interval", "Global_Interval", "Filename"])
        df_metadata_to_save.to_pickle("features\\"+metadata_filename)

        print(metadata_filename, "--- Saved Correctly")
        print("##########\n")


    print("Processing:", file)

    x = df.iloc[:, 1:].values
    y = df.iloc[:, 1].values


    if patID != last_patID:
        last_patID = patID
        global_interval = {0:0, 1:0}
        last_type = None
        data = []
        df_metadata = []

    elif last_file != None and file != last_file:
        if x[0][0] == last_type:
            global_interval[last_type] += 1


    for row in x:

        row_data = []
        row_metadata = []

        row_type = row[0] # Seizure or normal
        row_interval = row[1] # Filename Interval

        if row_type != last_type:

            if last_type != None:
                global_interval[last_type] += 1

            last_type = row_type

        for column in row[2:]:
            features = (np.mean(column), np.std(column), kurtosis(column), 
                        skew(column), entropy(column), np.min(column), np.max(column), np.sum(np.power(column,2))) 

            features = (np.max(column))
            row_data.append(features)

        metadata = (row_type, row_interval, global_interval[row_type], file)
        
        df_metadata.append(metadata)
        data.append(row_data)
    
    last_file = file

    

df_to_save = pd.DataFrame(data, columns=list(df.columns[3:]))
        
sampled = int(file.split("Sampled_")[1].split("-")[0])
win_size = int(file.split("WinSize_")[1].split("-")[0])
window = int(win_size/sampled)

filename = patID+"_classicfeatures_seizure_EEGwindow_"+str(window)+".pkl"
df_to_save.to_pickle("features\\"+filename)
print(filename, "--- Saved Correctly")


