import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split


# Read train and test csv files
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')
train_full = pd.read_csv('data/train.csv')
test_full = pd.read_csv('data/test.csv')

'''
Translate str to int
'''
def age_in_days(x):
    if x[1] in 'days': return int(x[0])
    if x[1] in 'weeks': return int(x[0]) * 7
    if x[1] in 'months': return int(x[0]) * 30
    if x[1] in 'years': return int(x[0]) * 365
    else: return int(x[0])

'''
Get list of unique breeds - without 'Mix' and '/'
'''
def list_breeds(df):
	un = df.Breed.unique()
	tmp = []
	for i in un:
		if 'Mix' in i:
			tmp.append(' '.join(i.split()[:-1]))
		elif '/' in i:
			tmp.append(i.split(sep='/')[0])
			tmp.append(i.split(sep='/')[1])
		else: tmp.append(i)
	return set(tmp)

'''
Get list of unique colors - without '/'
'''
def list_colors(df):
	colors = set()
	for i in df['Color'].str.replace('/', ' ').tolist():
		for j in i.split():
			colors.add(j)
	return colors

breed_tr = list_breeds(df_train)
breed_test = list_breeds(df_test)
breeds = breed_tr.union(breed_test)

colors = list_colors(df_train)


'''
input: initial dataframe (train or test)
output: changed dataframe with needed data preparation
'''
def data_preparation(df, is_train):
    df['Name'] = df['Name'].isna().apply(lambda x: 0 if x else 1) 
    
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df['Year'] = df['DateTime'].map(lambda x: x.year)
    df['Month'] = df['DateTime'].map(lambda x: x.month)
    df['dayOfWeek'] = df['DateTime'].map(lambda x: x.dayofweek)
    
    if is_train:
        df['OutcomeType'] = df['OutcomeType'].map({'Return_to_owner':0, 
                                           'Euthanasia':1, 'Adoption':2, 
                                           'Transfer':3, 'Died':4})
        df = df.drop(['AnimalID','OutcomeSubtype'], axis=1)
    
    df.AnimalType = df.AnimalType.map({'Cat': 0, 'Dog': 1})
    
    top = df['SexuponOutcome'].describe()['top'] # Change 'Unknown' to the most frequent
    nan_indexes = df[df.SexuponOutcome.isnull()].index
    df.loc[nan_indexes, 'SexuponOutcome'] = pd.Series(top, index=nan_indexes)
    df['Is_steril_sex'] = df['SexuponOutcome'].map({'Neutered Male':1, 
                                                 'Spayed Female':1, 'Intact Male':0, 
                                                 'Intact Female':0, 'Unknown':-1})
    df['SexuponOutcome'] = df['SexuponOutcome'].map({'Neutered Male':1, 
                                                 'Spayed Female':0, 'Intact Male':1, 
                                                 'Intact Female':0, 'Unknown':-1})
    
    df['AgeuponOutcome'] = df['AgeuponOutcome'].fillna('0 -1')
    df['AgeuponOutcome'] = df['AgeuponOutcome'].str.split().apply(age_in_days)
    df['AgeType'] = df['AgeuponOutcome'].apply(lambda x: 0 if (x // 365) < 1 else ( 1 if (x // 365) < 5 else 2) )
    #df['Young'] = df['AgeuponOutcome'].apply(lambda x: 1 if (x // 365) <= 1 else 0)
    #df['Adult'] = df['AgeuponOutcome'].apply(lambda x: 1 if (x // 365) > 1 and (x // 365) <= 5 else 0)
    #df['Old'] = df['AgeuponOutcome'].apply(lambda x: 1 if (x // 365) > 5 else 0)
    
    df['Mix'] = df['Breed'].apply(lambda x: 1 if 'Mix' in x else (1 if '/' in x else 0))
    
    for i in breeds:
        df[i] = df.Breed.apply(lambda x: 1 if i in x else 0)
    
    for i in colors:
        df[i] = df.Color.apply(lambda x: 1 if i in x else 0)
    #df.Color = df.Color.apply(lambda x: type_color(x))    
    #for i in group_color.keys():
        #df[i] = df.Color.apply(lambda x: 1 if x == i else 0)
    
    return df.drop(['DateTime', 'Breed', 'Color', 'AgeuponOutcome', 'SexuponOutcome'], axis=1)

# Final datafeames
train = data_preparation(df_train, 1)
test = data_preparation(df_test, 0)
test = test.drop('ID', axis = 1)

# Split by train and test
y = train['OutcomeType']
X = train.drop('OutcomeType', axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(X, y , test_size=0.7)


#Random Forest
rf = RandomForestClassifier(random_state=42, n_estimators=250, oob_score=True)
rf.fit(X_train,y_train)
print(rf.oob_score_ , log_loss(y_valid, rf.predict_proba(X_valid)))


#Logistic Reg
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', C=0.1)
logit.fit(X_train,y_train)
print(log_loss(y_valid, logit.predict_proba(X_valid)))

# Submission
logit = LogisticRegression()
logit.fit(X,y)
pred = logit.predict_proba(test)

ypred_submit = pd.DataFrame(pred)
submission = pd.DataFrame()
submission["ID"] = df_test['ID'].values
submission["Adoption"]= ypred_submit[2]
submission["Died"]= ypred_submit[4]
submission["Euthanasia"]= ypred_submit[1]
submission["Return_to_owner"]= ypred_submit[0]
submission["Transfer"]= ypred_submit[3]
submission = pd.DataFrame(submission)

submission.to_csv("submission01.csv",index=False)

