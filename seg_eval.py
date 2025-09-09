import numpy as np
import json
import pandas as pd
from sklearn.metrics import accuracy_score

with open("out_dict.json","r") as f:
	out_dict = json.load(f)

#for keys in out_dict:
	#print(keys, " history: \n", out_dict[keys]['history'],"\n")
	#print(keys," reason: \n", out_dict[keys]['reason'],"\n")
	#print(keys," comparison: \n", out_dict[keys]['comparison'],"\n")
	#print(keys," findings: \n", out_dict[keys]['findings'],"\n")
	#print(keys," nodule present?: \n", out_dict[keys]['nodule_present'],"\n")
	#print(keys," nodule size: \n", out_dict[keys]['nodule_size'],"\n")

df = pd.read_excel("labels.ods", engine="odf")

#Convert bools to strings
def normalize_bool(x):
	if isinstance(x, str):
		return x.strip().lower()
	return str(x).lower()

df["Nodule?"] = df["Nodule?"] #.apply(normalize_bool)
df[">5mm?"] = df[">5mm?"] #.apply(normalize_bool) #this says 5mm but we actually check for 6. Just a dumb typo by me
df["No cancer in history"] = df["No cancer in history"] #.apply(normalize_bool)

#print('df: ', df)
#print('out_dict: ', out_dict)

nod_acc=0.
size_acc=0.
hist_acc=0.
for subject in df["Subject"]:
	subject_str = str(subject)
	if subject > 1030:
		continue
	y_true_nodule = df[df["Subject"]==subject]["Nodule?"].iloc[0]
	y_pred_nodule = out_dict[subject_str]["nodule_present"]
	y_true_size = df[df["Subject"]==subject][">5mm?"].iloc[0]
	y_pred_size = out_dict[subject_str]["nodule_size"]
	y_true_hist = df[df["Subject"]==subject]["No cancer in history"].iloc[0]
	y_pred_hist = out_dict[subject_str]["history"]
	if (bool(y_true_nodule) == bool(y_pred_nodule)):
		nod_acc += 1.
	if (bool(y_true_size) == bool(y_pred_size)):
		size_acc += 1.
	if (bool(y_true_hist) == bool(y_pred_hist)):
		hist_acc += 1.

nod_acc /= 30
size_acc /= 30
#hist_acc /= 30

print("Nodule detection accuracy: ",nod_acc)
print("Size detection accuracy ", size_acc)
#print(hist_acc)
