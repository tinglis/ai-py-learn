# The purpose of this program is to compare various classifiers availible in the scikit-learn module. THe comparison is preformed on a small hard coded data set.


import numpy
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn import neighbors
from sklearn import neural_network
from sklearn import svm
from sklearn import gaussian_process
from sklearn import ensemble
from sklearn import discriminant_analysis

# CHALLENGE - create 3 more classifiers...
#1
#2
#3

print ' ----------------------------------------------'

#[hight,weight, shoe_size]

X = [[181,80,44], [177,70,43], [160,60,38], [154,54,37], [166, 65, 40],
	 [190,90,47], [175,64,39], 
	 [177,70,40], [159,55,37], [171,75,42], [181,85,43]]

Y = ['male', 'male', 'female', 'female', 'male',
	 'male', 'female', 
	 'female', 'female', 'male', 'male']

#print X
#print Y

#Decision Tree

clf = tree.DecisionTreeClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'Decision Tree'
print prediction

#Nearest Neighbours

clf = neighbors.KNeighborsClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'Nearest Neighbours'
print prediction

# neural network

clf = neural_network.MLPClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'neural network'
print prediction

#Gaussian Process

clf = gaussian_process.GaussianProcessClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'Gaussian Process'
print prediction

#SVM

clf = svm.SVC()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'SVM'
print prediction

# Random Forest

clf = ensemble.RandomForestClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'Random Forest'
print prediction

#AdaBoost

clf = ensemble.AdaBoostClassifier()	

clf = clf.fit(X, Y)

prediction = clf.predict([[190,70,43]])

print 'AdaBoost'
print prediction


print '--------------------------------------------'
