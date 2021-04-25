import pandas as pd
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OrdinalEncoder
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



def soil():
    traindf = pd.read_csv('datasets/soilDataset/train_timeseries.csv', sep=',', header=0)
    testdf = pd.read_csv('datasets/soilDataset/test_timeseries.csv', sep=',', header=0)

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
    
    return trainFeatures, trainTarget, testFeatures, testTarget


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
    return trainFeatures, trainTarget, testFeatures, testTarget

def uber():
    originaldf = pd.read_csv('datasets/uberDataset/uber_peru_2010.csv', sep=';', header=0)
    originaldf = originaldf.dropna(subset=['rider_score'])
    del originaldf['journey_id']   
    
    originaldf = originaldf.dropna(axis=0)
    originaldf = pd.DataFrame(data=originaldf)
    
    enc = OrdinalEncoder()
    enc.fit(originaldf)
    originaldf =  enc.transform(originaldf)
    
    originaldf = pd.DataFrame(data=originaldf)
    traindf = originaldf.sample(frac=0.75, random_state=0)
    testdf = originaldf.drop(traindf.index)  
    
    trainFeatures = traindf.iloc[:,:26]
    trainTarget = pd.DataFrame( traindf.iloc[:,26])
     
    testFeatures = testdf.iloc[:,:26]
    testTarget = pd.DataFrame( testdf.iloc[:,26])

    return trainFeatures, trainTarget, testFeatures, testTarget



def c_cars():
    originaldf = pd.read_csv('datasets/carsDataset/used_cars_data.csv', sep=',', header=0, nrows=10000, usecols=['back_legroom', 'bed', 'bed_length', 'body_type', 'city', 'city_fuel_economy', 'daysonmarket', 'dealer_zip', 'engine_cylinders', 'engine_displacement', 'engine_type', 'exterior_color', 'fleet', 'frame_damaged', 'franchise_dealer', 'franchise_make', 'front_legroom', 'fuel_tank_volume', 'fuel_type', 'has_accidents', 'height', 'highway_fuel_economy',
                             'horsepower', 'interior_color', 'isCab', 'is_cpo', 'is_new', 'is_oemcpo', 'length', 'listing_color', 'make_name', 'maximum_seating', 'mileage', 'model_name', 'owner_count', 'power', 'price', 'salvage', 'savings_amount', 'seller_rating', 'sp_id', 'theft_title', 'torque', 'transmission', 'transmission_display', 'trim_name', 'wheel_system', 'wheel_system_display', 'wheelbase', 'width', 'year'])


    le = preprocessing.LabelEncoder()
    le.fit(originaldf['bed'])
    tempDF = le.transform(originaldf['bed'])
    originaldf['bed'] = tempDF

    le.fit(originaldf['bed_length'])
    tempDF = le.transform(originaldf['bed_length'])
    originaldf['bed_length'] = tempDF

    le.fit(originaldf['back_legroom'])
    tempDF = le.transform(originaldf['back_legroom'])
    originaldf['back_legroom'] = tempDF

    le.fit(originaldf['body_type'])
    tempDF = le.transform(originaldf['body_type'])
    originaldf['body_type'] = tempDF

    le.fit(originaldf['city'])
    tempDF = le.transform(originaldf['city'])
    originaldf['city'] = tempDF

    le.fit(originaldf['city_fuel_economy'])
    tempDF = le.transform(originaldf['city_fuel_economy'])
    originaldf['city_fuel_economy'] = tempDF

    le.fit(originaldf['engine_cylinders'])
    tempDF = le.transform(originaldf['engine_cylinders'])
    originaldf['engine_cylinders'] = tempDF

    le.fit(originaldf['engine_type'])
    tempDF = le.transform(originaldf['engine_type'])
    originaldf['engine_type'] = tempDF

    le.fit(originaldf['exterior_color'])
    tempDF = le.transform(originaldf['exterior_color'])
    originaldf['exterior_color'] = tempDF   
        
    le.fit(originaldf['fleet'])
    tempDF = le.transform(originaldf['fleet'])
    originaldf['fleet'] = tempDF           
        
    le.fit(originaldf['frame_damaged'])
    tempDF = le.transform(originaldf['frame_damaged'])
    originaldf['frame_damaged'] = tempDF   
        
    le.fit(originaldf['franchise_dealer'])
    tempDF = le.transform(originaldf['franchise_dealer'])
    originaldf['franchise_dealer'] = tempDF   
       
    le.fit(originaldf['franchise_make'])
    tempDF = le.transform(originaldf['franchise_make'])
    originaldf['franchise_make'] = tempDF   
      
    le.fit(originaldf['front_legroom'])
    tempDF = le.transform(originaldf['front_legroom'])
    originaldf['front_legroom'] = tempDF   
      
    le.fit(originaldf['fuel_tank_volume'])
    tempDF = le.transform(originaldf['fuel_tank_volume'])
    originaldf['fuel_tank_volume'] = tempDF   
       
    le.fit(originaldf['fuel_type'])
    tempDF = le.transform(originaldf['fuel_type'])
    originaldf['fuel_type'] = tempDF  
             
    le.fit(originaldf['has_accidents'])
    tempDF = le.transform(originaldf['has_accidents'])
    originaldf['has_accidents'] = tempDF           
   
    le.fit(originaldf['height'])
    tempDF = le.transform(originaldf['height'])
    originaldf['height'] = tempDF           
    
    le.fit(originaldf['interior_color'])
    tempDF = le.transform(originaldf['interior_color'])   
    originaldf['interior_color'] = tempDF           
    
    le.fit(originaldf['isCab'])
    tempDF = le.transform(originaldf['isCab']) 
    originaldf['isCab'] = tempDF           
    
    le.fit(originaldf['is_cpo'])
    tempDF = le.transform(originaldf['is_cpo'])
    originaldf['is_cpo'] = tempDF           

    le.fit(originaldf['is_new'])
    tempDF = le.transform(originaldf['is_new'])
    originaldf['is_new'] = tempDF           
    
    le.fit(originaldf['is_oemcpo'])
    tempDF = le.transform(originaldf['is_oemcpo'])
    originaldf['is_oemcpo'] = tempDF           
    
    le.fit(originaldf['length'])
    tempDF = le.transform(originaldf['length'])
    originaldf['length'] = tempDF           
    
    le.fit(originaldf['listing_color'])
    tempDF = le.transform(originaldf['listing_color'])
    originaldf['listing_color'] = tempDF           
    
    le.fit(originaldf['make_name'])
    tempDF = le.transform(originaldf['make_name'])    
    originaldf['make_name'] = tempDF             
    
    le.fit(originaldf['maximum_seating'])
    tempDF = le.transform(originaldf['maximum_seating'])  
    originaldf['maximum_seating'] = tempDF  
                
    le.fit(originaldf['model_name'])
    tempDF = le.transform(originaldf['model_name'])   
    originaldf['model_name'] = tempDF  
             
    le.fit(originaldf['owner_count'])
    tempDF = le.transform(originaldf['owner_count'])
    originaldf['owner_count'] = tempDF
               
    le.fit(originaldf['power'])
    tempDF = le.transform(originaldf['power'])
    originaldf['power'] = tempDF 
              
    le.fit(originaldf['salvage'])
    tempDF = le.transform(originaldf['salvage'])
    originaldf['salvage'] = tempDF    
      
    le.fit(originaldf['theft_title'])
    tempDF = le.transform(originaldf['theft_title'])
    originaldf['theft_title'] = tempDF    

    le.fit(originaldf['torque'])
    tempDF = le.transform(originaldf['torque'])
    originaldf['torque'] = tempDF  
      
    le.fit(originaldf['transmission'])
    tempDF = le.transform(originaldf['transmission'])
    originaldf['transmission'] = tempDF  
      
    le.fit(originaldf['transmission_display'])
    tempDF = le.transform(originaldf['transmission_display'])
    originaldf['transmission_display'] = tempDF  
      
    le.fit(originaldf['trim_name'])
    tempDF = le.transform(originaldf['trim_name'])
    originaldf['trim_name'] = tempDF  
      
    le.fit(originaldf['wheel_system'])
    tempDF = le.transform(originaldf['wheel_system'])
    originaldf['wheel_system'] = tempDF    
    
    le.fit(originaldf['wheel_system_display'])
    tempDF = le.transform(originaldf['wheel_system_display'])
    originaldf['wheel_system_display'] = tempDF    
    
    le.fit(originaldf['wheelbase'])
    tempDF = le.transform(originaldf['wheelbase'])
    originaldf['wheelbase'] = tempDF    
    
    le.fit(originaldf['width'])
    tempDF = le.transform(originaldf['width'])
    originaldf['width'] = tempDF    
    
    
    
    # remplacer toute valeur NAN
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(originaldf)
    tempSeries = imp.transform(originaldf)

    tempDF = pd.DataFrame(data=tempSeries, columns=originaldf.columns)
    originaldf['engine_displacement']= tempDF['engine_displacement']
    originaldf['highway_fuel_economy']= tempDF['highway_fuel_economy']
    originaldf['horsepower']= tempDF['horsepower']
    originaldf['mileage']= tempDF['mileage']
    originaldf['seller_rating']= tempDF['seller_rating']
    
    traindf = originaldf.sample(frac=0.75, random_state=0)
    testdf = originaldf.drop(traindf.index)  
    
    trainFeatures = traindf  
    trainFeatures = trainFeatures.drop(columns=['price'], axis=1)
    trainTarget = pd.DataFrame( traindf['price'])
     
    testFeatures = traindf
    testFeatures = testFeatures.drop(columns=['price'], axis=1)
    testTarget = pd.DataFrame( testdf['price'])
    
    
    # print(originaldf)  

    # get numericals out of strings                   lambda x: str.extract(r'(\d)') if x[0] ==REGEX else np.nan'''if x[0] == "35.1 in" else np.nan'''
    # originaldf["back_legroom"] = originaldf.apply(   lambda x:  x[0].extract(r'(\dd)') , axis=1)
    
    # print(originaldf)
    # s = pd.Series(tempSeries[0])
    # s = s.str.extract(r'(\d)')
    # print (s)

    # imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
    # imp.fit(originaldf)
    # tempDF = imp.transform(originaldf)
    
    # tempDF = pd.DataFrame(data=tempDF)

    # originaldf.iloc[:,5] = tempDF.iloc[:,5] 
   # clf = tree.DecisionTreeClassifier()
   # clf = clf.fit(trainFeatures, trainTarget)
  #  print(clf.score(testFeatures, testTarget))
    
    gfg_csv_data = trainFeatures.to_csv('df.csv', index = True)

cars()