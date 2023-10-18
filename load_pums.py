import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
import os
os.chdir("/your path to ACS_PUMS data folder/ACS_PUMS")
import folktables
from folktables import ACSDataSource

def get_default_meta():
    features=[ 'AGEP','SCHL', 'MAR','DIS','ESP','CIT','MIG','MIL', 'ANC','NATIVITY','DEAR',
        'DEYE','DREM','SEX','RAC1P','PUMA','ST', 'OCCP','JWTRNS', 'POWPUMA']
    tasks=['Employment','Income','HealthInsurance','TravelTime','IncomePovertyRatio']
    
    return features,tasks

def load_data(mode='train',year='2018', state_list=['CA']):
    if int(year)>2018:        
        features=[ 'AGEP','SCHL', 'MAR','DIS','ESP','CIT','MIG','MIL', 'ANC','NATIVITY','DEAR',
        'DEYE','DREM','SEX','RAC1P','PUMA','ST', 'OCCP','JWTRNS', 'POWPUMA'] 
            
    else:    
        year='2018'
        features=[ 'AGEP','SCHL', 'MAR','DIS','ESP','CIT','MIG','MIL', 'ANC','NATIVITY','DEAR',
        'DEYE','DREM','SEX','RAC1P','PUMA','ST', 'OCCP','JWTR',#for 2018 data the feature is 'JWTR' which changed to 'JWTRNS' from 2019#
        'POWPUMA']
    data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person')
    acs_data = data_source.get_data( states=state_list,download=True)
    Employment = folktables.BasicProblem(
                             features=features,
                            target='ESR',
                            target_transform=lambda x: x == 1,
                            group='SEX',
                            preprocess=folktables.acs.adult_filter,
                            postprocess=lambda x: np.nan_to_num(x, -1),)
    Income = folktables.BasicProblem(
                         features=features,
                        target='PINCP',
                        target_transform=lambda x: x > 50000,
                        group='SEX',
                        preprocess=folktables.acs.adult_filter,
                        postprocess=lambda x: np.nan_to_num(x, -1),
                    )
    HealthInsurance = folktables.BasicProblem(
                         features=features,
                        target='HINS2',
                        target_transform=lambda x: x == 1,
                        group='SEX',
                        preprocess=folktables.acs.adult_filter,
                        postprocess=lambda x: np.nan_to_num(x, -1),
                    )
    
    TravelTime = folktables.BasicProblem(
                                 features=features,
                                target="JWMNP",
                                target_transform=lambda x: x > 20,
                                group='SEX',
                                preprocess=folktables.acs.adult_filter,
                                postprocess=lambda x: np.nan_to_num(x, -1),
                            )
    IncomePovertyRatio = folktables.BasicProblem(
                        features=features,
                        target='POVPIP',
                        target_transform=lambda x: x < 250,
                        group='SEX',
                        preprocess=folktables.acs.adult_filter,
                        postprocess=lambda x: np.nan_to_num(x, -1),
                    )
    f, l1, g = Employment.df_to_numpy(acs_data)
    f, l2, g = Income.df_to_numpy(acs_data)

    f, l3, g = HealthInsurance.df_to_numpy(acs_data)
    f, l4, g = TravelTime.df_to_numpy(acs_data)
    f, l5, g = IncomePovertyRatio.df_to_numpy(acs_data)
    
    y=np.array([[0 if v==False else 1 for v in l1],[0 if v==False else 1 for v in l2],[0 if v==False else 1 for v in l3],\
           [0 if v==False else 1 for v in l4],[0 if v==False else 1 for v in l5]])
    if mode!='test':
        ids=np.arange(len(f))
        X_train, X_val,in_tr,in_val  = train_test_split(f,ids, test_size=0.3,random_state=9)
        y_train,y_v=[y[i][in_tr] for i in range(len(y))],[y[i][in_val] for i in range(len(y))]
        g_train=g[in_tr]
        g_val=g[in_val]
        N_tasks=len(y)
        y_train=[torch.tensor(y_train[i]) for i in range(N_tasks)]
        
        return X_train, X_val,in_tr,in_val,y_train,y_v,g_train,g_val
    else:
        return f,y,g