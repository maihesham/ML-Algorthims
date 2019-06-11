import numpy as np
import pandas as pd
import random
from sklearn.model_selection import cross_val_score


def randomInt (low , high) :
    return random.randint(low,high)

def randomFloat (low , high):
    return random.uniform(low, high)
#load and prepare the data 
dataset = pd.read_csv('Social_Network_Ads.csv')
features = dataset.iloc[:,1:-1].values
goal = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
features[:,0] = encoder.fit_transform(features[:,0])

# standrize the values
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = scaler.fit_transform(features)

# split into train and test
from sklearn.cross_validation import train_test_split
train_set, test_set, goal_train, goal_test = train_test_split(features,goal,train_size =0.7,random_state=0)


#KNN ALGORITHM
from sklearn.neighbors import KNeighborsClassifier
k_value = randomInt(1,10)
classifier = KNeighborsClassifier(n_neighbors = k_value)
classifier.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, classifier.predict(test_set))
print("knn confusion_matrix is => ")
print(pd.DataFrame(cm))

#svc algorithm
# Applying k-Fold Cross Validation
print("knn accuracies : ")
accuracies = cross_val_score(estimator = classifier, X = train_set, y = goal_train, cv = 10)
print(accuracies)

#leogistic Regression
from sklearn.linear_model import LogisticRegression
itr = randomInt(100,1000)
t = randomFloat(00001,0.01)
logistic = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
          intercept_scaling=1, max_iter=itr, multi_class='ovr', n_jobs=1,
          penalty='l2', random_state=0, solver='liblinear', tol=t,
          verbose=0, warm_start=False)
logistic.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(goal_test, logistic.predict(test_set))
print("leogistic confusion_matrix is => ")
print(pd.DataFrame(cm))

#legositic alg
# Applying k-Fold Cross Validation
print("legositic accuracies : ")
logisticaccuracies = cross_val_score(estimator = logistic, X = train_set, y = goal_train, cv = 10)
print (logisticaccuracies)

#svc
from sklearn.svm import SVC
cvsclassifier = SVC(kernel ='linear', random_state = 0)
cvsclassifier.fit(train_set, goal_train)
from sklearn.metrics import confusion_matrix
# claclte num of errors in matrix
cm = confusion_matrix(goal_test, cvsclassifier.predict(test_set))
print("SVC confusion_matrix is => ")
print(pd.DataFrame(cm))
# Applying k-Fold Cross Validation
print("SVC accuracies : ")
svcaccuracies = cross_val_score(estimator = cvsclassifier, X = train_set, y = goal_train, cv = 10)
print (svcaccuracies)
