import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import tree
from c45 import c_stroke , c_soil, c_uber
from adaboost import a_stroke, a_soil, a_uber
from bagging import b_soil, b_stroke, b_uber


print ("c soil")
c_soil()

# print ("c stroke")
# c_stroke()

# print ("c uber")
# c_uber()

# print ("a soil")
# a_soil()

# print ("a stroke")
# a_stroke()

# print ("a uber")
# a_uber()

# print ("b soil")
# b_soil()

# print ("b stroke")
# b_stroke()

# print ("b uber")
# b_uber()