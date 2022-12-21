
import json
import numpy as np
import os
import pandas as pd

# Prints and creates metric from file
f = open("Transfomer_LOO_metrics_balanced.json")
data = json.load(f)

accuracy = []
precision_0 = []
precision_1 = []
recall_0 = []
recall_1 = []
f1_score_0 = []
f1_score_1 = []


# Leave one out
values = list(data.keys())
values = values[:-6] # or -7, depends on the file
print("Keys:", values, "Check if everything is in there")

# Create pd df to convert to excel
df = []

for value in values:
    sublist = []
    sublist.append(round(data[value]["accuracy"], 4))
    sublist.append(round(data[value]["precision"][0], 4))
    sublist.append(round(data[value]["precision"][1], 4))
    sublist.append(round(data[value]["recall"][0], 4))
    sublist.append(round(data[value]["recall"][1], 4))
    sublist.append(round(data[value]["f1-score"][0], 4))
    sublist.append(round(data[value]["f1-score"][1], 4))
    df.append(sublist)

df = pd.DataFrame(df, columns = ["accuracy", "precision_0", "precision_1", "recall_0", "recall_1", "f1-score_0", "f1-score_1"])
df.to_excel("metriques.xlsx")

for value in values:
    accuracy.append(data[value]["accuracy"])
    precision_0.append(data[value]["precision"][0])
    precision_1.append(data[value]["precision"][1])
    recall_0.append(data[value]["recall"][0])
    recall_1.append(data[value]["recall"][1])
    f1_score_0.append(data[value]["f1-score"][0])
    f1_score_1.append(data[value]["f1-score"][1])



accuracy = np.array(accuracy)
precision_0 = np.array(precision_0)
precision_1 = np.array(precision_1)
recall_0 = np.array(recall_0)
recall_1 = np.array(recall_1)
f1_score_0 = np.array(f1_score_0)
f1_score_1 = np.array(f1_score_1)


print("%0.4f accuracy with a standard deviation of %0.4f" % (accuracy.mean(), accuracy.std()))
print("%0.4f precision 0 with a standard deviation of %0.4f" % (precision_0.mean(), precision_0.std()))
print("%0.4f precision 1 with a standard deviation of %0.4f" % (precision_1.mean(), precision_1.std()))
print("%0.4f recall 0 with a standard deviation of %0.4f" % (recall_0.mean(), recall_0.std()))
print("%0.4f recall 1 with a standard deviation of %0.4f" % (recall_1.mean(), recall_1.std()))
print("%0.4f f1_score 0 with a standard deviation of %0.4f" % (f1_score_0.mean(), f1_score_0.std()))
print("%0.4f f1_score 1 with a standard deviation of %0.4f" % (f1_score_1.mean(), f1_score_1.std()))
