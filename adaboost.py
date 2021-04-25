from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
from preprocessing import soil, stroke, uber


def a_soil():   
  
    trainFeatures, trainTarget, testFeatures, testTarget = soil()
   
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainFeatures, trainTarget)
    
    print(clf.score(testFeatures, testTarget))

    


def a_stroke():
    trainFeatures, trainTarget, testFeatures, testTarget = stroke()
 
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainFeatures, trainTarget)

    print(clf.score(testFeatures, testTarget))
    
def a_uber():
    trainFeatures, trainTarget, testFeatures, testTarget = uber()

    
    clf = AdaBoostClassifier(n_estimators=100, random_state=0)
    clf.fit(trainFeatures, trainTarget)

    print(clf.score(testFeatures, testTarget))
