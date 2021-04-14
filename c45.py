import sklearn as sk
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn import tree
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer




def soil():   
    traindf = pd.read_csv('datasets/soilDataset/train_timeseries.csv', sep=',', header=0)
    testdf = pd.read_csv('datasets/soilDataset/test_timeseries.csv', sep=',', header=0  )

    del traindf['date']
    traindf = traindf.dropna(axis=0)

    trainFeatures = traindf.iloc[:,:19]
    trainTarget = pd.DataFrame( traindf.iloc[:,19])
    trainTarget = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(trainTarget)
    trainTarget = np.ravel(trainTarget)



    del testdf['date']
    testdf = testdf.dropna(axis=0)

    testFeatures = testdf.iloc[:,:19]
    testTarget = pd.DataFrame( testdf.iloc[:,19])
    testTarget = KBinsDiscretizer(n_bins=3, encode="ordinal").fit_transform(testTarget)
    testTarget = np.ravel(testTarget)

    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget) 
    print (clf.score(testFeatures, testTarget))
    
    
def stroke():   
    originaldf = pd.read_csv('datasets/strokeDataset/healthcare-dataset-stroke-data.csv', sep=',', header=0)
    
    traindf = originaldf.sample(frac=0.75, random_state=0)
    testdf = originaldf.drop(traindf.index)
    

    del traindf['id']
    traindf = traindf.dropna(axis=0)

    trainFeatures = traindf.iloc[:,:10]
    trainTarget = pd.DataFrame( traindf.iloc[:,10])
    enc = OrdinalEncoder()
    enc.fit(trainFeatures)

    trainFeatures =  enc.transform(trainFeatures)

    del testdf['id']
    testdf = testdf.dropna(axis=0)

    testFeatures = testdf.iloc[:,:10]
    testTarget = pd.DataFrame( testdf.iloc[:,10])
    enc2 = OrdinalEncoder()
    enc2.fit(testFeatures)
    testFeatures =  enc2.transform(testFeatures)

    
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget) 
    print (clf.score(testFeatures, testTarget))

def uber():
    originaldf = pd.read_csv('datasets/uberDataset/uber_peru_2010.csv', sep=';', header=0)
    originaldf = originaldf.dropna(subset=['rider_score'])
    del originaldf['journey_id']   
    
    originaldf = originaldf.dropna(axis=0)
 
    
    #remplacer toute valeur NAN
    # imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=-1)
    # imp.fit(originaldf)
    # originaldf = imp.transform(originaldf)

    #reconvertir vers pd.dataframe
    originaldf = pd.DataFrame(data=originaldf)

    #encoder les données string
    #L'erreur survient lorsque les données NAN imputées sont un INT -1 et que les données originales sont des strings
    #Encoders require their input to be uniformly strings or numbers
    enc = OrdinalEncoder()
    enc.fit(originaldf)
    originaldf =  enc.transform(originaldf)
    
    
    originaldf = pd.DataFrame(data=originaldf)
    print (originaldf)
    traindf = originaldf.sample(frac=0.75, random_state=0)
    testdf = originaldf.drop(traindf.index)
    
    
    
    trainFeatures = traindf.iloc[:,:26]
    trainTarget = pd.DataFrame( traindf.iloc[:,26])
    
     
    testFeatures = testdf.iloc[:,:26]
    testTarget = pd.DataFrame( testdf.iloc[:,26])
    
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(trainFeatures, trainTarget) 
    print (clf.score(testFeatures, testTarget))

def cars():
    originaldf = pd.read_csv('datasets/carsDataset/used_cars_data.csv', sep=',', header=0, nrows=10000)    
    del originaldf['is_certified']   
    del originaldf['vehicle_damage_category']   
    del originaldf['combine_fuel_economy']  
    del originaldf['trimId'] 
    del originaldf['bed_height'] 
    del originaldf['description'] 
    del originaldf['cabin'] 
    del originaldf['vin'] 
    
     #remplacer toute valeur NAN
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    imp.fit(originaldf)
    tempDF = imp.transform(originaldf)
    #print(originaldf.iloc[:,2])
    originaldf.iloc[:,1:3] = tempDF[:,1:3]

    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(originaldf)
    tempDF = imp.transform(originaldf)
    
    tempDF = pd.DataFrame(data=tempDF)

    originaldf.iloc[:,5] = tempDF.iloc[:,5] 
    
    #print(originaldf.to_string())
    print (originaldf) 
    gfg_csv_data = originaldf.to_csv('df.csv', index = True)

cars()