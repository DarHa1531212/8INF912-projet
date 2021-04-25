from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
import numpy as np
import pandas as pd
from preprocessing import soil, stroke, uber


def b_soil():   
    trainFeatures, trainTarget, testFeatures, testTarget = soil()
   
    clf = BaggingClassifier(base_estimator=SVC(),
                n_estimators=10, random_state=0).fit(trainFeatures, trainTarget)

    print(clf.score(testFeatures, testTarget))
    

      
def b_stroke(): 
    trainFeatures, trainTarget, testFeatures, testTarget = stroke()

    clf = BaggingClassifier(base_estimator=SVC(),
                            n_estimators=10, random_state=0).fit(trainFeatures, trainTarget)

    print(clf.score(testFeatures, testTarget))


def b_uber():
    trainFeatures, trainTarget, testFeatures, testTarget = uber()

    
    clf = BaggingClassifier(base_estimator=SVC(),
                            n_estimators=10, random_state=0).fit(trainFeatures, trainTarget)

    print(clf.score(testFeatures, testTarget))
