import numpy as np
import pandas as pd
import urllib.request as ur


import sklearn
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

from sklearn import metrics
from sklearn.metrics import accuracy_score


url="https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
raw_data = ur.urlopen(url)
dataset = np.loadtxt(raw_data, delimiter=',')
print(dataset[0])


X = dataset[:, 0:48]
y = dataset[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=17)