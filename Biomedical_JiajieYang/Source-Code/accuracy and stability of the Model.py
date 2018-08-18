import pylab as pl
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import math
import random
from sklearn.externals import joblib

with open('E:\python\maintask\maintask7\Training.csv', 'r', ) as f:
	reader = csv.reader(f)
	rows1 = [row for row in reader]
	column = [row[1] for row in reader]
	a=len(rows1)
rows1=np.array(rows1)


X=rows1[:,1:len(rows1[0])-2]
y=rows1[:,len(rows1[0])-1]
print(len(X))
X=X.astype(float)
y=y.astype(float)



for i in range(len(X)-1,-1,-1):
	if float(np.max(X[i])) < 50:
		X=np.delete(X,i,axis=0)
		y=np.delete(y,i,axis=0)

print(len(X))

a=list(range(len(X)))
slice = random.sample(a, len(a))
XX1=X[slice[1]]
yy1=y[slice[1]]
XX2=XX1
yy2=yy1
for j in range(2,len(slice)):
	if j<0.8*len(a):
		XX1=np.row_stack((XX1, X[slice[j]]))
		yy1=np.row_stack((yy1, y[slice[j]]))
	else:
		XX2=np.row_stack((XX2, X[slice[j]]))
		yy2=np.row_stack((yy2, y[slice[j]]))

print(len(XX1),len(XX1[0]))
print(len(yy1))

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=10, max_features='auto', max_leaf_nodes=None,
            min_impurity_split=1e-07, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=10, n_jobs=1, oob_score=True, random_state=10,
            verbose=0, warm_start=False)
clf.fit(XX1, yy1)



for i in range(0,len(XX2)):
	if clf.predict_proba([XX2[i]])[0,0]<0.15 or clf.predict_proba([XX2[i]])[0,0]>0.85:
		XX1=np.row_stack((XX1, XX2[i]))
		yy1=np.insert(yy1, len(yy1), values=clf.predict([XX2[i]]), axis=0)
clf.fit(XX1, yy1)
joblib.dump(clf, 'clf.pkl')

count=0
for i in range(0,len(XX2)):
	if clf.predict([XX2[i]])==yy2[i]:
		count=count+1
	print(clf.predict([XX2[i]]), yy2[i])
if count/len(yy2)*100>50:
	print(count/len(yy2)*100)
	print(count,len(yy2))

#l=clf.feature_importances_
#print(l)
#print(len(l))