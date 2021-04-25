import sklearn as sk
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from preprocessing import soil, stroke, uber


def c_soil():

    trainFeatures, trainTarget, testFeatures, testTarget = soil()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget)
    print(clf.score(testFeatures, testTarget))


def c_stroke():
    trainFeatures, trainTarget, testFeatures, testTarget = stroke()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget)
    print(clf.score(testFeatures, testTarget))


def c_uber():

    trainFeatures, trainTarget, testFeatures, testTarget = uber()

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget)
    print(clf.score(testFeatures, testTarget))

