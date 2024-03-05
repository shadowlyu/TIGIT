import pandas as pd
import numpy as np
import csv
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import warnings
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pickle

f1  = open('decoys-selected.smi','r')

decoys_list = []
for lines in f1:
    _list = [lines]
    decoys_list.extend(_list)


df1 = pd.DataFrame()
df1['smiles'] = decoys_list

class FP:
    
    def __init__(self, fp, names):
        self.fp = fp
        self.names = names
#    def __str__(self):
#        return "%d bit FP" % len(self.fp)
    def __len__(self):
        return len(self.fp)

def get_cfps(mol, radius=4, nBits=2048, useFeatures=False, counts=False, dtype=np.float32):
    arr = np.zeros((1,), dtype)
    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures, bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return arr

def calFP(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = get_cfps(mol)
        return fp
    except Exception as e:
        return None
		
df1['FP'] = df1['smiles'].apply(calFP)
df1['label'] = '0'

file = "ippidb_compounds.csv"
df2 = pd.read_csv(file)
df3 = pd.DataFrame()
df3['smiles'] = df2['canonical_smile']
df3['FP'] = df3['smiles'].apply(calFP)
df3['label'] = '1'

df5 = pd.concat([df1,df3], ignore_index=True)

x = np.vstack((df5['FP']))
y = np.array(df5['label'].astype('int8'))

seed = 5290  
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=seed)
for tr_idx, te_idx in sss.split(x, y):
    x_train, y_train = x[tr_idx], y[tr_idx]
    x_test, y_test = x[te_idx], y[te_idx]
	
eval_set = [(x_train, y_train), (x_test, y_test)]
params = {
    'n_estimators': 221,
    'objective': 'binary:logistic',
    'gamma': 0.4,
    'max_depth': 9,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 1,
    'seed': 5290,
    'nthread': -1,
    'tree_method': 'gpu_hist',
    'scale_pos_weight': 1,
    'reg_lambda': 1,
    'reg_alpha': 0.1,
}


xgb1 = XGBClassifier(**params)

xgb1.fit(x_train,y_train,early_stopping_rounds=15, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

y_pred = xgb1.predict(x_test)
y_pred_proba = xgb1.predict_proba(x_test)
predictions = [round(value) for value in y_pred]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)


pickle.dump(xgb1, open("pima.pickle.dat", "wb"))

results = xgb1.evals_result()

epochs = len(results['validation_0']['error'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['error'], label='Train')
ax.plot(x_axis, results['validation_1']['error'], label='Test')
ax.legend()
plt.ylabel('Classification Error')
plt.title('XGBoost Classification Error')