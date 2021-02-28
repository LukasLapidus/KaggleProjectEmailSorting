# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Loading data
train_df = pd.read_csv('train_ml.csv', index_col=0)
test_df = pd.read_csv('test_ml.csv', index_col=0)


# Name of the targets
outputs = ['updates','personal','promotions','forums','purchases','travel','spam','social']


# We remove any occurance in the trainning data without a defined target
train_df.dropna(axis=0, subset=outputs, inplace=True)


# Separating the inputs and outputs
y = train_df[outputs]
X = train_df.copy()
X.drop(outputs, axis=1, inplace=True)


# Test data
X_test = test_df.copy()



# Correcting the 'mail_type' column

def correct_mail_type(Xd):
    Xd['mail_type'] = Xd['mail_type'].mask( Xd['mail_type']=='Multipart/Mixed' , 'multipart/mixed')
    Xd['mail_type'] = Xd['mail_type'].mask( Xd['mail_type']=='Multipart/Alternative' , 'multipart/alternative')
    Xd['mail_type'] = Xd['mail_type'].mask( Xd['mail_type'].isin(["text/HTML", "text/html ", "Text/Html"]) , 'text/html')
    return Xd

X = correct_mail_type(X)
X_test = correct_mail_type(X_test)

#print(X['mail_type'].unique())
#print(_testX['mail_type'].unique())



def utiliser_date(df):          # Using the feature date
    
    df_date = df[['date']]      # We transform the dataframe into an array
    date = np.array(df_date)[:,0]

    seconde = []
    minute = []
    heure = []
    day = []
    annee = []
    month = []
    
    for mot in date :
        M = mot.split()
        jour = 'None'
        mois = 'None'
        decade = 'None'
        
        for m in M:
            if ":" in m and len(m) == 8:
                horaire = m
            elif m in ['Mon,', 'Tue,', 'Wed,', 'Thu,', 'Fri,', 'Sat,', 'Sun,']:
                jour = m[:-1]
            elif "20" in m and len(m) == 4 :
                decade = int(m)
            elif m in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']:
                mois = m
        
        if mot != []:
            h = int(horaire[0:2])
            m = int(horaire[3:5])
            s = int(horaire[6:8])
        else :
            s, m, h = 'None','None','None'
            
        day.append(jour)
        seconde.append(s)
        minute.append(m)
        heure.append(h)
        month.append(mois)
        annee.append(decade)
    
    return(np.array(heure), np.array(minute),np.array(seconde), np.array(day), np.array(annee), np.array(month))



# We create the new data from the date
# Infos_date = heure, minute, seconde, day, annee, mois

cols_date = ['heure', 'minute', 'seconde', 'day', 'annee', 'mois']

Infos_date_X = utiliser_date(X)
Infos_date_X_test = utiliser_date(X_test)

for info in range(6):
    X[cols_date[info]] = pd.DataFrame(Infos_date_X[info])
    X[cols_date[info]] = X[cols_date[info]].mask( X[cols_date[info]]=='None' , np.nan)
    X_test[cols_date[info]] = pd.DataFrame(Infos_date_X_test[info])
    X_test[cols_date[info]] = X_test[cols_date[info]].mask( X_test[cols_date[info]]=='None' , np.nan)


X.drop('date',axis=1,inplace=True)
X_test.drop('date',axis=1,inplace=True)






# Separation of the train data

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)


# Working on the data

# Differntitation of the type of data (we decide to only use one hot encoding)
cols_num = X_train.select_dtypes(exclude=['object']).columns
cols_label = []
cols_onehot = ['day','mois','mail_type','annee','org','tld']



# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')


# Preprocessing for categorical data with label
label_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('label', LabelEncoder())
])


# Preprocessing for categorical data with onehot
onehot_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])





# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, cols_num),
#    ('lab', label_transformer, cols_label),
    ('one', onehot_transformer, cols_onehot)
])



# Creating the pipeline with the whole process
my_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', RandomForestClassifier(n_estimators=100, random_state=0))
                             ])


# Preprocessing of training data, fit model 
my_pipeline.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
preds = my_pipeline.predict(X_valid)


# Evaluate the model
score = mean_absolute_error(y_valid, preds)
acc = accuracy_score(y_valid,preds)
print('MAE:', score)
print('acc:', acc)


""" Cross validation unuseful because of the size of the dataset
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("MAE scores:\n", scores.mean())
"""


my_pipeline.fit(X,y)
preds_test = my_pipeline.predict_proba(X_test)
preds_test = np.array([preds_test[i][:,1] for i in range(8)]).T


# Save test predictions to file

output = pd.DataFrame(preds_test, columns=['updates', 'personal', 'promotions','forums', 'purchases', 'travel','spam', 'social'])
output.to_csv("submission_KaggleEmail.csv", index=True, index_label='Id')





