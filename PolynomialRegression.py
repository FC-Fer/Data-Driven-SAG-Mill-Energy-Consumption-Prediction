import numpy as np
import csv
import random
import time
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ##################
# Preprocessing
# ##################

def Extract_data(DB_SolarMining, IndVar, Target):
    N_features_input = len(IndVar)
    Train_data_in = np.zeros((len(DB_SolarMining[:,0]), N_features_input))
    Train_data_out = np.zeros((len(DB_SolarMining[:,0]), 1))
    for i in range(len(DB_SolarMining[:,0])):
        for ind0 in range(N_features_input):
            Train_data_in[i,ind0] = DB_SolarMining[i:i+1,IndVar[ind0]]
        Train_data_out[i,:] = DB_SolarMining[i,Target]
    np.savetxt('Test.txt',Train_data_in)
    return Train_data_in, Train_data_out

# #####################
def Fit_PolynomialRegression(degree, IndVar, Target, Stats):
    #print("Polynomial degree:", degree)
    
    # Parameters
    #N_features_input = len(IndVar)
    #N_categ = 1

    # Importing Data Base
    DB_SolarMining = np.loadtxt("Trainv2.txt") #Modify (Train data normalized)
    DB_Testing = np.loadtxt("Valv2v2.txt") #Modify (Validation data normalized)
    DB_Training = DB_SolarMining
     
    # Preprocessing
    Train_data_in, Train_data_out = Extract_data(DB_SolarMining, IndVar, Target)     
    Test_data_in, Test_data_out = Extract_data(DB_Testing, IndVar, Target) 
   
    #N_step = int(Test_data_in.shape[0])
        
    "Creates a polynomial regression model for the given degree"

    poly_features = PolynomialFeatures(degree=degree)

    # transforms the existing features to higher degree features.
    X_train_poly = poly_features.fit_transform(Train_data_in)

    # fit the transformed features to Linear Regression
    poly_model = LinearRegression()
    poly_model.fit(X_train_poly, Train_data_out)

    # predicting on training data-set
    y_train_predicted = poly_model.predict(X_train_poly)

    # predicting on test data-set
    y_test_predict = poly_model.predict(poly_features.fit_transform(Test_data_in))
    #y_test_predict = poly_model.predict(Test_data_in)
    

    # evaluating the model on training dataset
    rmse_train = np.sqrt(mean_squared_error(Train_data_out, y_train_predicted))
    r2_train = r2_score(Train_data_out, y_train_predicted)
    coefcorr_train = np.corrcoef(Train_data_out[:,0], y_train_predicted[:,0])
    #print ("rmseTrain :", rmse_train)
    #print ("r2Train :" ,r2_train)
    #print ("CCTrain :" ,coefcorr_train)

    # evaluating the model on test dataset
    rmse_test = np.sqrt(mean_squared_error(Test_data_out, y_test_predict))
    r2_test = r2_score(Test_data_out, y_test_predict)
    coefcorr_test = np.corrcoef(Test_data_out[:,0], y_test_predict[:,0])
    #print ("rmseTest :" ,rmse_test)
    #print ("r2Test :", r2_test)
    #print ("CCTest :", coefcorr_test)

    
    a = 19114 #Modify (target average in train data)
    b = int(Stats)
    print("T:",Target," a:",a," b:",b)
    # Preprocessing
    Num_Elements = int(Test_data_in.shape[0])
    Results = np.zeros((Num_Elements,4)) # day, real, estimated, diff
    for ind1 in range(Num_Elements):
        Sim_in_real = Test_data_in[ind1:(ind1+1),:]
        Sim_out_real = Test_data_out[ind1:(ind1+1),:]
        One_pred = poly_model.predict(poly_features.fit_transform(Sim_in_real))
        #One_pred = poly_model.predict(Sim_in_real)        
        Real = Sim_out_real[0]
        #print("Actual consumption [%.3f] vs Predicted consumption: [%.3f]" % (Real, One_pred))
        Results[ind1,0], Results[ind1,1], Results[ind1,2], Results[ind1,3] = (ind1+1), int(Real*b+a), int(One_pred*b+a), (int(Real*b+a) - int(One_pred*b+a))
    np.savetxt('Models_1_3_6_9_10/Sim_Res_Validation_%s_%s.txt'%(degree,Target), Results, fmt='%3.4e', delimiter='\t', newline='\n')      
    
    return None #print("Fitting model done")



# "t" Parameters
#0: Date
#1: Feed Tonnage [Tph]	    2: Percentage under 2 inch [%] 3: Spindle Speed [%[]  
#4: Power consumption [kWh] 5: Weight [t]                  6: Bearing Pressure [Psi]	
#7: Solid Percentage [%]    8: Water [m3/h]	
#9: DeltaFT [Tph]          10: DeltaSSp [Rpm]
# "t+1" parameters 
#11: P t5min [kW]  12: P t_10min	13: P t_20min	14: P t_30min	15: P t_1h
StatsDsv = [3252] # EC 0.5, EC 1, EC 2, EC 4, EC 8 #Modify (targets dsv. in train data)
DegreeVec = np.linspace(1,6,6) #Modify if necessary
#TargetVec = [9,10,11,12,13] 
TargetVec = [14] #Modify if necesarry (target columns)
for ind_0 in range(len(TargetVec)):  
    for ind_1 in range(len(DegreeVec)):
        IndVar = [1, 3, 6, 9, 10]    #Modify if necessary (Input columns)
        Target = int(TargetVec[ind_0]) 
        Degree = int(DegreeVec[ind_1])
        print("Target: ", Target, "Degree; ", Degree, "\n")
        Fit_PolynomialRegression(degree=Degree, IndVar=IndVar, Target=Target, Stats=StatsDsv[ind_0])

