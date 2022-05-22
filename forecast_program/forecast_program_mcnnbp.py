# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 17:11:42 2021

@author: steve
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
from keras import regularizers
from keras.models import load_model
from keras.initializers import RandomNormal
from keras.layers import Convolution1D,Dense,concatenate,RepeatVector,LSTM,Lambda, MaxPooling1D
from keras.layers.core import Activation,Flatten,Reshape
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam_v2
from keras.layers import BatchNormalization
#%%

remain_index=pd.read_csv("D:/research/pm25_3days_output/2019-2021/remain_index.csv")
stimulate_input=pd.read_excel("D:/research/pm25_3days_output/2019-2021/EPA&simulate_dataset(new).xlsx",sheet_name="forecast")
real_output=pd.read_excel("D:/research/pm25_3days_output/2019-2021/72hours_real_stimulate_pm25.xlsx",sheet_name="real")
complete_data=pd.read_csv("D:/research/pm25_3days_output/2019-2021/complete_data(for_convo_use)).txt")
# complete_data.iloc[632784,6]=np.mean(complete_data.iloc[:,6])
# complete_data.iloc[536149,7]=np.mean(complete_data.iloc[:,7])
find_all_index=pd.read_csv("D:/research/pm25_3days_output/2019-2021/find_all_index.csv").iloc[:,1]

#%%
interpolate=stimulate_input.iloc[:,3:]
interpolate[interpolate<0]=np.nan
mean=interpolate.mean(axis=0).reset_index(drop=True)
for i in range(0,len(interpolate)):
    for j in range(0,len(interpolate.iloc[i])):
        if math.isnan(interpolate.iloc[i,j]):
            stimulate_input.iloc[i,j+3]=mean.iloc[j]
        if interpolate.iloc[i,j]<0:   
            stimulate_input.iloc[i,j+3]=mean.iloc[j]

interpolate=complete_data.iloc[:,4:]
interpolate[interpolate<0]=np.nan
mean=interpolate.mean(axis=0).reset_index(drop=True)
for i in range(0,len(interpolate)):
    for j in range(0,len(interpolate.iloc[i])):
        if math.isnan(interpolate.iloc[i,j]):
            complete_data.iloc[i,j+4]=mean.iloc[j]
        if interpolate.iloc[i,j]<0:   
            complete_data.iloc[i,j+4]=mean.iloc[j]

#%%
# data normalization
#scaler
scaler = MinMaxScaler().fit(complete_data.iloc[:,4:])
scaler_2 = MinMaxScaler().fit(stimulate_input.iloc[remain_index.iloc[:,1],3:])
scaler_3 = MinMaxScaler().fit(real_output.iloc[:,2:])
#transform data
epa_data_norm =scaler.transform(complete_data.iloc[:,4:])
stimulate_input_norm = scaler_2.fit_transform(stimulate_input.iloc[remain_index.iloc[:,1],3:])
real_output_norm = scaler_3.fit_transform(real_output.iloc[:,2:])

#%%

convo_input=([[]*1 for i in range(0,len(find_all_index))])
for i in range(0,len(find_all_index)):
    for j in range(-23,1):
        convo_input[i].append(np.asarray(complete_data.iloc[find_all_index.iloc[i]+j,4:]))
        # convo_input[i].append(np.asarray(epa_data_norm[find_all_index.iloc[i]+j,:]))


for i in range(0,len(convo_input)):
    for j in range(0,len(convo_input[i])):
        for k in range(0,len(convo_input[i][j])):
            convo_input[i][j][k]=float(convo_input[i][j][k])

#%%
            
input_info=(stimulate_input.iloc[remain_index.iloc[:,1],1:3]).reset_index(drop=True)
# input_one=stimulate_input_norm
input_one=np.asarray(stimulate_input.iloc[remain_index.iloc[:,1],3:])


# model_output=real_output_norm[remain_index.iloc[:,1],:]
model_output=np.asarray(real_output.iloc[remain_index.iloc[:,1],2:])

input_two=[]
for i in range(0,len(remain_index)):
    input_two.append(convo_input[remain_index.iloc[i,1]])

#%%
    
#shuffle data run this
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

#build hybrid model
K.clear_session() 
inputs1 = Input(shape=(73,))
inputs2 = Input(shape=(24,8))

output= Lambda(lambda x: tf.expand_dims(x, -1))(inputs1)
output=Convolution1D(filters=36,kernel_size=3,input_shape=(73,1),padding='same')(output)
output=BatchNormalization()(output)
output=Activation('relu')(output) 
output=Convolution1D(filters=36,kernel_size=3,padding='same')(output)
output=BatchNormalization()(output)
output=Activation('relu')(output) 
output=Flatten()(output)

output2=Convolution1D(filters=24,kernel_size=3,input_shape=(24,8),padding='same')(inputs2)
output2=BatchNormalization()(output2)
output2=Activation('relu')(output2) 
output2=Convolution1D(filters=24,kernel_size=3,padding='same')(output2)
output2=BatchNormalization()(output2)
output2=Activation('relu')(output2) 
output2=Flatten()(output2)

merge_output=concatenate([output,output2],axis=-1) 
final_output=Dense(36)(merge_output)     
final_output=Activation('relu')(final_output) 
final_output=Dense(72)(final_output)

model = Model(inputs=[inputs1,inputs2], outputs=final_output)
learning_rate=1e-3
adam = adam_v2.Adam(lr=learning_rate)
model.compile(optimizer=adam,loss="mae")
earlystopper = EarlyStopping(monitor='val_loss', patience=32, verbose=0)        
save_path="D:/research/pm25_3days_output/2019-2021/hybrid_cnn-cnn.hdf5"
checkpoint =ModelCheckpoint(save_path,save_best_only=True)
callback_list=[earlystopper,checkpoint]        

model.fit([training_random_input_one,training_random_input_two], training_random_output, epochs=50, batch_size=64,validation_split=0.1,callbacks=callback_list)

#%%
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

writer = pd.ExcelWriter('F:/my_paper/pm25_3days/3rd time summit(ES&T)/feature_extraction/feature_extraction_result(training)2.xlsx', engine='xlsxwriter')
pd.DataFrame(layer_output.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_one")
pd.DataFrame(layer_output_2.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_two")
pd.DataFrame(layer_2_output.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_one")
pd.DataFrame(layer_2_output_2.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_two")
pd.DataFrame(testing_random_output).to_excel(writer,sheet_name="observe_dataset")
writer.save()

#%%

#hybrid model forecasting
model=load_model(save_path, custom_objects={'tf': tf}) 

#shuffled
pred_train=(model.predict([training_random_input_one,training_random_input_two],batch_size=64))
pred_test=(model.predict([testing_random_input_one,testing_random_input_two],batch_size=64))


#%%

#denormalization
# pred_train=scaler_3.inverse_transform(pred_train)
# pred_test=scaler_3.inverse_transform(pred_test)

# training_random_output=scaler_3.inverse_transform(training_random_output)
# testing_random_output=scaler_3.inverse_transform(testing_random_output)

#%%
#calculate R2 and save into excel

os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")
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

writer = pd.ExcelWriter('D:/research/pm25_3days_output/2019-2021/hybrid_input/hybrid_model(cnn-cnn)-performance2.xlsx', engine='xlsxwriter')
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
pred_test=(model.predict([testing_input_one,testing_input_two],batch_size=64))


#%%
get_1st_layer_output=K.function([model.layers[0].input],[model.layers[7].output])
layer_output = get_1st_layer_output([testing_input_one])[0]
layer_output = np.squeeze(layer_output)
get_2nd_layer_output=K.function([model.layers[0].input],[model.layers[13].output])
layer_2_output = get_2nd_layer_output([testing_input_one])[0]
layer_2_output = np.squeeze(layer_2_output)

get_1st_layer_output_2=K.function([model.layers[2].input],[model.layers[8].output])
layer_output_2 = get_1st_layer_output_2([testing_input_two])[0]  
layer_output_2 = np.squeeze(layer_output_2)
get_2nd_layer_output_2=K.function([model.layers[2].input],[model.layers[14].output])
layer_2_output_2 = get_2nd_layer_output_2([testing_input_two])[0]  
layer_2_output_2 = np.squeeze(layer_2_output_2)

writer = pd.ExcelWriter('F:/my_paper/pm25_3days/3rd time summit(ES&T)/feature_extraction/feature_extraction_result(testing)2.xlsx', engine='xlsxwriter')
pd.DataFrame(layer_output.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_one")
pd.DataFrame(layer_output_2.mean(axis=2)).to_excel(writer,sheet_name="1stlayer_input_two")
pd.DataFrame(layer_2_output.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_one")
pd.DataFrame(layer_2_output_2.mean(axis=2)).to_excel(writer,sheet_name="2ndlayer_input_two")
pd.DataFrame(testing_input_info).to_excel(writer,sheet_name="observe_datetime")
pd.DataFrame(testing_output).to_excel(writer,sheet_name="observe_dataset")
writer.save()
#%%

#calculate R2 and save into excel

os.chdir("D:/important/work/PM2.5_competition/one_year/new_version(8input)/new_model")
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

writer = pd.ExcelWriter('D:/research/pm25_3days_output/2019-2021/hybrid_input/hybrid_model(cnn-cnn)-testing_performance2.xlsx', engine='xlsxwriter')
testing_R2.to_excel(writer,sheet_name="testing-R2")
testing_RMSE.to_excel(writer,sheet_name="testing-RMSE")
testing_mape.to_excel(writer,sheet_name="testing-mape")


forecast_result=pd.concat([testing_input_info.reset_index(drop=True),pd.DataFrame(pred_test)],axis=1).reset_index(drop=True)
forecast_result.to_excel(writer,sheet_name="forecast")
model_real_output=pd.DataFrame(testing_output).reset_index(drop=True)
model_real_output.to_excel(writer,sheet_name="realoutput")
# Close the Pandas Excel writer and output the Excel file.
writer.save()
