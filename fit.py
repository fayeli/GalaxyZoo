# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 10:11:12 2014

@author: manc
"""

import numpy as np
import pandas as pd
from sklearn import tree

colors = pd.read_csv('colors.csv',header=0,index_col=0)
solutions = pd.read_csv('training_solutions.csv',index_col=0)
roundness = pd.read_csv('roundness.csv',header=0,index_col=0)
hueStds = pd.read_csv('hueStds.csv',header=0,index_col=0)

#size of training/test set
setsize = 3000
colors = colors.iloc[:2*setsize]
solutions = solutions.iloc[:2*setsize]
roundness = roundness.iloc[:2*setsize]

data_train = colors.iloc[:setsize].join(roundness[:setsize]).join(hueStds[:setsize])
data_test = colors.iloc[setsize:2*setsize].join(roundness[setsize:2*setsize]).join(hueStds[setsize:2*setsize])

error_train = pd.Series(index=data_train.index,data=0)
error_test = pd.Series(index=data_test.index,data=0)

n = len(solutions.columns) #no. of columns

name = np.empty(n,dtype='|S10') #string type, 10 places
train_error = np.empty(n)
test_error = np.empty(n)

solutions = np.around(solutions)
for i in range(n):
    col = solutions.columns[i]
    target = solutions[col]
    target_train = target.iloc[0:setsize]
    target_test = target.iloc[setsize:2*setsize]
    clf = tree.DecisionTreeClassifier(min_samples_split=50)
    fitted = clf.fit(data_train,target_train)
    #for errors regarding each column
    name[i] = col
    train_error[i] = 100.0 * sum(fitted.predict(data_train)!=target_train)/len(data_train)
    test_error[i] = 100.0 * sum(fitted.predict(data_test)!=target_test)/len(data_test)
    #for errors about each row
    error_train = error_train + np.abs(fitted.predict(data_train)-target_train)
    error_test = error_test +  np.abs(fitted.predict(data_test)-target_test)

#general distribution of number of errors
#attribute and class, and then error rate for that class

print('Training error for each attribute to classify:')
print(train_error)
print('Generalization error for each attribute to classify:')
print(test_error)


'''
#show pictures with the most errors
error_train.sort(ascending=True)
print(error_train.head())
error_test.sort(ascending=False)
print(error_test.head())
'''

'''
errors_dict = {'Column Name': name, 'Training Error': train_error, 'Test Error': test_error}
errors = pd.DataFrame(errors_dict)
errors.to_csv('column_errors.csv')
'''