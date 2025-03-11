# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:49:08 2022

@author: steve
"""

import pandas as pd
import numpy as np
import os

path='D:/lab/research/pm25_3days_output/github'
os.chdir(path)

#%%
mcnnbp_trainval=pd.read_excel("performance(mcnn-bp).xlsx",sheet_name="forecast")
cnnlstm_trainval=pd.read_excel("performance(mcnn-bp).xlsx",sheet_name="forecast")
trainval_output=pd.read_excel("performance(mcnn-bp).xlsx",sheet_name="realoutput")

shuffle_index=mcnnbp_trainval.iloc[:,1]
cnnbp_trainval=pd.read_excel("performance(cnn-bp).xlsx",sheet_name="forecast").sort_values(by=[0], ascending=True)
cnnbp_trainval=cnnbp_trainval.reset_index(drop=True); cnnbp_trainval=cnnbp_trainval.iloc[shuffle_index,:].reset_index(drop=True)

#%%
mcnnbp_test=pd.read_excel("testing_performance(mcnn-bp).xlsx",sheet_name="forecast")
cnnlstm_test=pd.read_excel("testing_performance(mcnn-bp).xlsx",sheet_name="forecast")
cnnbp_test=pd.read_excel("testing_performance(cnn-bp).xlsx",sheet_name="forecast")
test_output=pd.read_excel("testing_performance(mcnn-bp).xlsx",sheet_name="realoutput")

#%%
def convert_daily(hourlydata):
    hourlydata=np.array(hourlydata)
    daily_data=[]
    daily_data.append(np.mean(hourlydata[:,0:24],axis=1))
    daily_data.append(np.mean(hourlydata[:,24:48],axis=1))
    daily_data.append(np.mean(hourlydata[:,48:72],axis=1))
    daily_data=np.transpose(np.array(daily_data)).astype('float')
    return daily_data

mcnnbp_train_daily=convert_daily(mcnnbp_trainval.iloc[0:40063,5:])
mcnnbp_val_daily=convert_daily(mcnnbp_trainval.iloc[40063:,5:])
mcnnbp_test_daily=convert_daily(mcnnbp_test.iloc[:,3:])

cnnbp_train_daily=convert_daily(cnnbp_trainval.iloc[0:40063,5:])
cnnbp_val_daily=convert_daily(cnnbp_trainval.iloc[40063:,5:])
cnnbp_test_daily=convert_daily(cnnbp_test.iloc[:,3:])

cnnlstm_train_daily=convert_daily(cnnlstm_trainval.iloc[0:40063,5:])
cnnlstm_val_daily=convert_daily(cnnlstm_trainval.iloc[40063:,5:])
cnnlstm_test_daily=convert_daily(cnnlstm_test.iloc[:,3:])
train_output_daily=convert_daily(trainval_output.iloc[0:40063,4:])
val_output_daily=convert_daily(trainval_output.iloc[40063:,4:])
test_output_daily=convert_daily(test_output.iloc[:,1:])

#%%
os.chdir("D:/lab/research/research_use_function")
from error_indicator import error_indicator

mcnnbp_train_R2=[]
cnnbp_train_R2=[]
cnnlstm_train_R2=[]

mcnnbp_val_R2=[]
cnnlstm_val_R2=[]
cnnbp_val_R2=[]

mcnnbp_test_R2=[]
cnnlstm_test_R2=[]
cnnbp_test_R2=[]

for i in range(0,3):
    mcnnbp_train_R2.append(error_indicator.np_R2(train_output_daily[:,i],mcnnbp_train_daily[:,i]))
    cnnlstm_train_R2.append(error_indicator.np_R2(train_output_daily[:,i],cnnlstm_train_daily[:,i]))
    cnnbp_train_R2.append(error_indicator.np_R2(train_output_daily[:,i],cnnbp_train_daily[:,i]))

    mcnnbp_val_R2.append(error_indicator.np_R2(val_output_daily[:,i],mcnnbp_val_daily[:,i]))
    cnnlstm_val_R2.append(error_indicator.np_R2(val_output_daily[:,i],cnnlstm_val_daily[:,i])) 
    cnnbp_val_R2.append(error_indicator.np_R2(val_output_daily[:,i],cnnbp_val_daily[:,i]))    
    
    mcnnbp_test_R2.append(error_indicator.np_R2(test_output_daily[:,i],mcnnbp_test_daily[:,i]))
    cnnlstm_test_R2.append(error_indicator.np_R2(test_output_daily[:,i],cnnlstm_test_daily[:,i]))
    cnnbp_test_R2.append(error_indicator.np_R2(test_output_daily[:,i],cnnbp_test_daily[:,i]))    

mcnnbp_train_RMSE=[]
cnnlstm_train_RMSE=[]
cnnbp_train_RMSE=[]

mcnnbp_val_RMSE=[]
cnnlstm_val_RMSE=[]
cnnbp_val_RMSE=[]

mcnnbp_test_RMSE=[]
cnnlstm_test_RMSE=[]
cnnbp_test_RMSE=[]

for i in range(0,3):
    mcnnbp_train_RMSE.append(error_indicator.np_RMSE(train_output_daily[:,i],mcnnbp_train_daily[:,i]))
    cnnlstm_train_RMSE.append(error_indicator.np_RMSE(train_output_daily[:,i],cnnlstm_train_daily[:,i]))
    cnnbp_train_RMSE.append(error_indicator.np_RMSE(train_output_daily[:,i],cnnbp_train_daily[:,i]))    
    
    mcnnbp_val_RMSE.append(error_indicator.np_RMSE(val_output_daily[:,i],mcnnbp_val_daily[:,i]))
    cnnlstm_val_RMSE.append(error_indicator.np_RMSE(val_output_daily[:,i],cnnlstm_val_daily[:,i]))    
    cnnbp_val_RMSE.append(error_indicator.np_RMSE(val_output_daily[:,i],cnnbp_val_daily[:,i])) 
    
    mcnnbp_test_RMSE.append(error_indicator.np_RMSE(test_output_daily[:,i],mcnnbp_test_daily[:,i]))
    cnnlstm_test_RMSE.append(error_indicator.np_RMSE(test_output_daily[:,i],cnnlstm_test_daily[:,i]))
    cnnbp_test_RMSE.append(error_indicator.np_RMSE(test_output_daily[:,i],cnnbp_test_daily[:,i]))
    
mcnnbp_train_mape=[]
cnnlstm_train_mape=[]
cnnbp_train_mape=[]

mcnnbp_val_mape=[]
cnnlstm_val_mape=[]
cnnbp_val_mape=[]

mcnnbp_test_mape=[]
cnnlstm_test_mape=[]
cnnbp_test_mape=[]

for i in range(0,3):
    mcnnbp_train_mape.append(error_indicator.np_mape(train_output_daily[:,i],mcnnbp_train_daily[:,i]))
    cnnlstm_train_mape.append(error_indicator.np_mape(train_output_daily[:,i],cnnlstm_train_daily[:,i]))
    cnnbp_train_mape.append(error_indicator.np_mape(train_output_daily[:,i],cnnbp_train_daily[:,i]))
    
    mcnnbp_val_mape.append(error_indicator.np_mape(val_output_daily[:,i],mcnnbp_val_daily[:,i]))
    cnnlstm_val_mape.append(error_indicator.np_mape(val_output_daily[:,i],cnnlstm_val_daily[:,i]))    
    cnnbp_val_mape.append(error_indicator.np_mape(val_output_daily[:,i],cnnbp_val_daily[:,i]))
    
    mcnnbp_test_mape.append(error_indicator.np_mape(test_output_daily[:,i],mcnnbp_test_daily[:,i]))
    cnnlstm_test_mape.append(error_indicator.np_mape(test_output_daily[:,i],cnnlstm_test_daily[:,i]))    
    cnnbp_test_mape.append(error_indicator.np_mape(test_output_daily[:,i],cnnbp_test_daily[:,i]))
    
#%%
simulate_data=pd.read_excel("D:/research/pm25_3days_output/2019-2021/EPA&simulate_dataset(new).xlsx",sheet_name="forecast")
remain_index=pd.read_csv("D:/research/pm25_3days_output/2019-2021/remain_index.csv")
simulate_output=np.asarray(simulate_data.iloc[remain_index.iloc[:,1],4:])  
real_output=pd.read_excel("D:/research/pm25_3days_output/2019-2021/72hours_real_stimulate_pm25.xlsx",sheet_name="real")
real_output=np.asarray(real_output.iloc[remain_index.iloc[:,1],2:])

#%%

real_train=convert_daily(real_output[0:40063,:])
real_val=convert_daily(real_output[40063:50079,:])
real_test=convert_daily(real_output[50079:,:])

simulate_train=convert_daily(simulate_output[0:40063,:])
simulate_val=convert_daily(simulate_output[40063:50079,:])
simulate_test=convert_daily(simulate_output[50079:,:])

simulate_train_R2=[];simulate_val_R2=[];simulate_test_R2=[]
simulate_train_RMSE=[];simulate_val_RMSE=[];simulate_test_RMSE=[]
simulate_train_mape=[];simulate_val_mape=[];simulate_test_mape=[]
for i in range(0,3):
    simulate_train_R2.append(error_indicator.np_R2(real_train[:,i],simulate_train[:,i]))
    simulate_val_R2.append(error_indicator.np_R2(real_val[:,i],simulate_val[:,i]))
    simulate_test_R2.append(error_indicator.np_R2(real_test[:,i],simulate_test[:,i]))
    simulate_train_RMSE.append(error_indicator.np_RMSE(real_train[:,i],simulate_train[:,i]))
    simulate_val_RMSE.append(error_indicator.np_RMSE(real_val[:,i],simulate_val[:,i]))
    simulate_test_RMSE.append(error_indicator.np_RMSE(real_test[:,i],simulate_test[:,i]))    
    simulate_train_mape.append(error_indicator.np_mape(real_train[:,i],simulate_train[:,i]))
    simulate_val_mape.append(error_indicator.np_mape(real_val[:,i],simulate_val[:,i]))
    simulate_test_mape.append(error_indicator.np_mape(real_test[:,i],simulate_test[:,i]))    

