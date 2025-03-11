# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 15:58:15 2022

@author: Steve
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import math
import numpy as np
import random
import os
import tensorflow as tf
from keras import Model
from keras.engine.input_layer import Input
from keras import backend as K
from keras.models import load_model
from keras.layers import Convolution1D,Dense,concatenate,Lambda
from keras.layers.core import Activation,Flatten
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2
from keras.layers import BatchNormalization
#%%
remain_index=pd.read_csv("D:/lab/research/pm25_3days_output/github/dataset/remain_index.csv")
simulate_input=pd.read_csv("D:/lab/research/pm25_3days_output/github/dataset/AS_simulate_input.csv")
real_output=pd.read_csv("D:/lab/research/pm25_3days_output/github/dataset/EPA_output.csv")
complete_data=pd.read_csv("D:/lab/research/pm25_3days_output/github/dataset/complete_data(for_convo_use)).txt")
find_all_index=pd.read_csv("D:/lab/research/pm25_3days_output/github/dataset/find_all_index.csv").iloc[:,1]

#%%
""" data interpolation """

os.chdir(r"D:\lab\research\research_use_function")
from preprocessing import preprocessing
preprocessing_module1=preprocessing(simulate_input.iloc[:,3:])
simulate_input_=preprocessing_module1.interpolate(remove_negative=True)
simulate_input.iloc[:,3:]=simulate_input_

preprocessing_module2=preprocessing(complete_data.iloc[:,4:])
complete_data_=preprocessing_module2.interpolate(remove_negative=True)
complete_data.iloc[:,4:]=complete_data_

#%%
""" data normalization """

#scaler
scaler = MinMaxScaler().fit(complete_data.iloc[:,4:])
scaler_2 = MinMaxScaler().fit(simulate_input.iloc[remain_index.iloc[:,1],3:])
scaler_3 = MinMaxScaler().fit(real_output.iloc[:,3:])
#transform data
epa_data_norm =scaler.transform(complete_data.iloc[:,4:])
stimulate_input_norm = scaler_2.fit_transform(simulate_input.iloc[remain_index.iloc[:,1],3:])
real_output_norm = scaler_3.fit_transform(real_output.iloc[:,3:])

#%%
""" select specific dataset as model input """

# select specific dataset
convo_input=([[]*1 for i in range(0,len(find_all_index))])
for i in range(0,len(find_all_index)):
    for j in range(-23,1):
        convo_input[i].append(np.asarray(complete_data.iloc[find_all_index.iloc[i]+j,4:]))

for i in range(0,len(convo_input)):
    for j in range(0,len(convo_input[i])):
        for k in range(0,len(convo_input[i][j])):
            convo_input[i][j][k]=float(convo_input[i][j][k])

# remove error dataset
input_info=(simulate_input.iloc[remain_index.iloc[:,1],1:3]).reset_index(drop=True)
input_one=np.asarray(simulate_input.iloc[remain_index.iloc[:,1],3:])
model_output=np.asarray(real_output.iloc[remain_index.iloc[:,1],4:])

input_two=[]
for i in range(0,len(remain_index)):
    input_two.append(convo_input[remain_index.iloc[i,1]])

#%%
""" Data shuffling """

def generate_random_list(list_length):
    random_list=list(range(0,list_length))
    random.shuffle(random_list)
    return np.asarray(random_list)
random_list=generate_random_list(50079)

random_input_info=input_info.iloc[random_list,:].reset_index(drop=True)
random_input_one=input_one[random_list,:]
random_output=model_output[random_list,:]

random_input_two=[[]*1 for i in range(0,len(random_list))]
for i in range(0,len(random_list)):
    random_input_two[i].append(input_two[random_list[i]])

training_random_input_one=random_input_one[0:int(len(random_input_one)*0.8),:]
training_random_input_two=np.asarray(random_input_two[0:int(len(random_input_one)*0.8)]).astype(np.float) 
training_random_input_two=training_random_input_two[:,0,:,:]
training_random_output=np.asarray(random_output[0:int(len(random_output)*0.8),:])

testing_random_input_one=random_input_one[int(len(random_input_one)*0.8):,:]
testing_random_input_two=np.asarray(random_input_two[int(len(random_input_one)*0.8):]).astype(np.float) 
testing_random_input_two=testing_random_input_two[:,0,:,:]
testing_random_output=np.asarray(random_output[int(len(random_output)*0.8):,:])
            
#%%
""" Build MCNN-BP """

K.clear_session() 
inputs2 = Input(shape=(24,8))

output= Lambda(lambda x: tf.expand_dims(x, -1))(inputs2)
output=Convolution1D(filters=36,kernel_size=3,input_shape=(73,1),padding='same')(output)
output=BatchNormalization()(output)
output=Activation('relu')(output) 
output=Convolution1D(filters=36,kernel_size=3,padding='same')(output)
output=BatchNormalization()(output)
output=Activation('relu')(output) 
output=Flatten()(output)

final_output=Dense(36)(output)     
final_output=Activation('relu')(final_output) 
final_output=Dense(72)(final_output)

model = Model(inputs=inputs2, outputs=final_output)
learning_rate=1e-3
adam = adam_v2.Adam(lr=learning_rate)
model.compile(optimizer=adam,loss="mae")
earlystopper = EarlyStopping(monitor='val_loss', patience=32, verbose=0)        
save_path="D:/lab/research/pm25_3days_output/github/cnn-bp.hdf5"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        

model.fit(training_random_input_two, training_random_output, epochs=50, batch_size=64,validation_split=0.1,callbacks=callback_list)

#%%
""" observe convolution layer output's feature """

get_1st_layer_output=K.function([model.layers[0].input],[model.layers[7].output])
layer_output = get_1st_layer_output([testing_random_input_one])[0]
layer_output = np.squeeze(layer_output)
get_2nd_layer_output=K.function([model.layers[0].input],[model.layers[13].output])
layer_2_output = get_1st_layer_output([testing_random_input_one])[0]
layer_2_output = np.squeeze(layer_2_output)

get_1st_layer_output_2=K.function([model.layers[2].input],[model.layers[8].output])
layer_output_2 = get_1st_layer_output_2([testing_random_input_two])[0]  
layer_output_2 = np.squeeze(layer_output_2)
get_2nd_layer_output_2=K.function([model.layers[2].input],[model.layers[14].output])
layer_2_output_2 = get_2nd_layer_output_2([testing_random_input_two])[0]  
layer_2_output_2 = np.squeeze(layer_2_output_2)

writer = pd.ExcelWriter('D:/lab/research/pm25_3days_output/github/feature_extraction_result(training).xlsx', engine='xlsxwriter')
pd.DataFrame(layer_output.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_one")
pd.DataFrame(layer_output_2.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_two")
pd.DataFrame(layer_2_output.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_one")
pd.DataFrame(layer_2_output_2.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_two")
pd.DataFrame(testing_random_output).to_excel(writer,sheet_name="observe_dataset")
writer.save()

#%%
""" Model forecasting """

#load model
model=load_model(save_path, custom_objects={'tf': tf}) 

#model predict
pred_train=(model.predict(training_random_input_two,batch_size=64))
pred_test=(model.predict(testing_random_input_two,batch_size=64))

#%%
""" Model Performances """

#calculate R2 and save into excel
os.chdir(r"D:\lab\research\research_use_function")
from error_indicator import error_indicator
training_R2=[];testing_R2=[]
training_RMSE=[];testing_RMSE=[]
training_mape=[];testing_mape=[]
#shuffled
for i in range(0,72):
    training_R2.append(error_indicator.np_R2(training_random_output[:,i],pred_train[:,i]))
    testing_R2.append(error_indicator.np_R2(testing_random_output[:,i],pred_test[:,i]))
    training_RMSE.append(error_indicator.np_RMSE(training_random_output[:,i],pred_train[:,i]))
    testing_RMSE.append(error_indicator.np_RMSE(testing_random_output[:,i],pred_test[:,i]))
    training_mape.append(error_indicator.np_mape(training_random_output[:,i],pred_train[:,i]))
    testing_mape.append(error_indicator.np_mape(testing_random_output[:,i],pred_test[:,i]))    
    
training_R2=pd.DataFrame(training_R2)
testing_R2=pd.DataFrame(testing_R2)   
training_RMSE=pd.DataFrame(training_RMSE)
testing_RMSE=pd.DataFrame(testing_RMSE)   
training_mape=pd.DataFrame(training_mape)
testing_mape=pd.DataFrame(testing_mape)   

writer = pd.ExcelWriter('D:/lab/research/pm25_3days_output/github/performance(cnn-bp).xlsx', engine='xlsxwriter')
training_R2.to_excel(writer,sheet_name="training-R2")
testing_R2.to_excel(writer,sheet_name="testing-R2")
training_RMSE.to_excel(writer,sheet_name="training-RMSE")
testing_RMSE.to_excel(writer,sheet_name="testing-RMSE")
training_mape.to_excel(writer,sheet_name="training-mape")
testing_mape.to_excel(writer,sheet_name="testing-mape")

training_testing_index=[]
for i in range(0,len(random_list)):
    if i<0.8* int(len(random_list))-1:
        training_testing_index.append(0)
    else:
        training_testing_index.append(1)
training_testing_index=pd.DataFrame(training_testing_index)
forecast_result=pd.concat([pd.DataFrame(pred_train),pd.DataFrame(pred_test)]).reset_index(drop=True)
final_forecast_result=pd.concat([pd.DataFrame(random_list),random_input_info],axis=1)
final_forecast_result=pd.concat([final_forecast_result,training_testing_index,forecast_result],axis=1)
final_forecast_result.to_excel(writer,sheet_name="forecast")
model_real_output=pd.concat([pd.DataFrame(training_random_output),pd.DataFrame(testing_random_output)]).reset_index(drop=True)
final_real_output=pd.concat([pd.DataFrame(random_list),random_input_info],axis=1)
final_real_output=pd.concat([final_real_output,model_real_output],axis=1)
final_real_output.to_excel(writer,sheet_name="realoutput")
# Close the Pandas Excel writer and output the Excel file.
writer.save()

#%%
""" Model forecasting (additional dataset) """

#hybrid model forecasting
model=load_model(save_path, custom_objects={'tf': tf}) 

testing_input_info=input_info.iloc[len(random_list):,:]
testing_input_one=np.asarray(input_one[len(random_list):,:])
testing_output=np.asarray(model_output[len(random_list):,:])

testing_input_two=[]
for i in range(0,63074-len(random_list)):
    testing_input_two.append(input_two[i+len(random_list)])
testing_input_two=np.asarray(testing_input_two).astype('float64')

#shuffled
pred_test=(model.predict(testing_input_two,batch_size=64))

#%%
""" Model Performances (additional dataset) """

#calculate R2 and save into excel

os.chdir(r"D:\lab\research\research_use_function")
from error_indicator import error_indicator
testing_R2=[];testing_RMSE=[];testing_mape=[]
#shuffled
for i in range(0,72):
    testing_R2.append(error_indicator.np_R2(testing_output[:,i],pred_test[:,i]))
    testing_RMSE.append(error_indicator.np_RMSE(testing_output[:,i],pred_test[:,i]))
    testing_mape.append(error_indicator.np_mape(testing_output[:,i],pred_test[:,i]))      

testing_R2=pd.DataFrame(testing_R2)   
testing_RMSE=pd.DataFrame(testing_RMSE)   
testing_mape=pd.DataFrame(testing_mape)   

writer = pd.ExcelWriter('D:/lab/research/pm25_3days_output/github/testing_performance(cnn-bp).xlsx', engine='xlsxwriter')
testing_R2.to_excel(writer,sheet_name="testing-R2")
testing_RMSE.to_excel(writer,sheet_name="testing-RMSE")
testing_mape.to_excel(writer,sheet_name="testing-mape")


forecast_result=pd.concat([testing_input_info.reset_index(drop=True),pd.DataFrame(pred_test)],axis=1).reset_index(drop=True)
forecast_result.to_excel(writer,sheet_name="forecast")
model_real_output=pd.DataFrame(testing_output).reset_index(drop=True)
model_real_output.to_excel(writer,sheet_name="realoutput")
# Close the Pandas Excel writer and output the Excel file.
writer.save()