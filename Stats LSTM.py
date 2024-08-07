import numpy as np

#0: Day	   1: FT [Tph]	   2: EC tn [kW]    3: BearingPressure [Psi]	4: SpindleSpeed [Rpm]	   5: Wtr [m3/h]	6: SPe [%]	
#7: DeltaFT [Tph]	8: DeltaSSp [Rpm]	9: EC tn+1 [kW]	      10: EC t_1hr	11: EC t_2hr	12: EC t_4hr	13: EC t_8hr
#14: FT05h	15: FT1h	16: FT2h	17: FT4h	18: FT8h


StatsDsv = [3148] # EC 0.5, EC 1, EC 2, EC 4, EC 8
#NumHiddenLayers = [4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120]
NumHiddenLayers = np.linspace(4,200,50)
CorrCoefMatrix = np.zeros((len(NumHiddenLayers),len(StatsDsv)+1))
CorrCoefMatrix[:,0] = NumHiddenLayers

rRMSE = np.zeros((len(StatsDsv),len(NumHiddenLayers)))
for ind_1 in range(len(NumHiddenLayers)):    
    for ind_0 in range(len(StatsDsv)):    
        n_hidden = int(NumHiddenLayers[ind_1])
        N_units = [n_hidden]
        NumRecurrent = len(N_units)        
        IndVar = [1, 3, 6, 9, 10]
        Target = [14]
        N_lag = 8
        training_iters = 51
        LocModel = 'Models_1_3_6_9_10/Lag%s_nh%sx%st_LSTM_V1_3_6_9_10_T%s_E%s'%(N_lag, n_hidden, NumRecurrent, Target, training_iters)  
        File = LocModel + '/Sim_Res_Testing.txt'
        ResFile = np.loadtxt(fname=File)
        #print(ResFile[:,3])
        temp_RMSE = 100*np.sqrt(np.sum((ResFile[:,1] - ResFile[:,2])**2)/len(ResFile[:,1]))/np.average(ResFile[:,1])
        #print("EC: ", ind_0+1, "NumHid: ", n_hidden,  "RMSE: ", 100*np.sqrt(np.sum((ResFile[12000:,1] - ResFile[12000:,2])**2)/len(ResFile[12000:,1]))/np.average(ResFile[12000:,1]))
        rRMSE[ind_0,ind_1] = temp_RMSE
        temp_CorrCoef = np.corrcoef(ResFile[:,1], ResFile[:,2])
        CorrCoefMatrix[ind_1,ind_0+1] = temp_CorrCoef[1,0]
        print("EC: ", ind_0+1, "NumHid: ", n_hidden,  "CorrCoef: ", temp_CorrCoef[1,0])
  
np.savetxt(fname="CorrCoefMatrix_LSTM.txt", X=CorrCoefMatrix, fmt='%.2f', delimiter='\t')
np.savetxt(fname="rRMSE_LSTM.txt", X=rRMSE, fmt='%.2f', delimiter='\t')
            
        
