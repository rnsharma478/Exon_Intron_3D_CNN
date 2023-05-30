#library dependencies
import pandas as pd
import numpy as np
import os
import gc
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.initializers import Constant
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

#reading the data
folder_path = "H:\\3_lac_intron_exon_data\\3lakh_intron_exon\\supervised_data"
sub_folder_list = os.listdir(folder_path)
folder_dict = {}

for sub_folder in sub_folder_list:
    path = folder_path + "\\"+ str(sub_folder)
    #exec(f'sub_{sub_folder} = os.listdir(path)') #makes a list all files in all sub - folders and assigns it to different variables
    folder_dict[sub_folder] = os.listdir(path)
    for file in folder_dict[sub_folder]:
        newname_file = file.replace('.csv', '_df')
        newname_file = newname_file.replace('IE_','')
        newname_file = newname_file.replace('_50','')
        print(newname_file)
        final_path = path + "\\" + str(file)
        exec(f'exon_{newname_file} = pd.read_csv(final_path, header=None)')
        gc.collect()
print("data is read")

#adding class to data
exon_start = [exon_bbone_START_df, exon_bp_START_df, exon_hbond_START_df, exon_inter_START_df, exon_intra_START_df, exon_solvation_START_df, exon_stack_START_df]
exon_end = [exon_bbone_END_df, exon_bp_END_df, exon_hbond_END_df, exon_inter_END_df, exon_intra_END_df, exon_solvation_END_df, exon_stack_END_df]
cds = [exon_bbone_CDS_df, exon_bp_CDS_df, exon_hbond_CDS_df, exon_inter_CDS_df, exon_intra_CDS_df, exon_solvation_CDS_df, exon_stack_CDS_df]

for df in exon_start:
    df["class"] = 1

for df in exon_end:
    df["class"] = 2

for df in cds:
    df["class"] = 0
print("class is added to all dataframes")

#restructing data for modelling
exon_bbone_df = pd.concat([exon_start[0], exon_end[0], cds[0]], ignore_index = True)
exon_bp_df = pd.concat([exon_start[1], exon_end[1], cds[1]], ignore_index = True)
exon_hbond_df = pd.concat([exon_start[2], exon_end[2], cds[2]], ignore_index = True)
exon_inter_df = pd.concat([exon_start[3], exon_end[3], cds[3]], ignore_index = True)
exon_intra_df = pd.concat([exon_start[4], exon_end[4], cds[4]], ignore_index = True)
exon_solvation_df = pd.concat([exon_start[5], exon_end[5], cds[5]], ignore_index = True)
exon_stack_df = pd.concat([exon_start[6], exon_end[6], cds[6]], ignore_index = True)
print("different dataframes with 8 lakh sequences and 50 positions with class in each processed")


Y = exon_bbone_df["class"] #we can take any one dataframe's class as all have same 
X = pd.concat([exon_bbone_df.drop(["class"], axis=1), exon_bp_df.drop(["class"],axis=1), exon_hbond_df.drop(["class"], axis=1), exon_inter_df.drop(["class"], axis=1), exon_intra_df.drop(["class"],axis=1), exon_solvation_df.drop(["class"], axis=1), exon_stack_df.drop(["class"], axis=1)], axis=1, ignore_index=True)

#split data into training and testing - further reshape it to input in model 
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True)
xtrain = np.array(x_train)
xtest = np.array(x_test)
print(‘train shape:’, xtrain.shape)
print(‘test shape:’, xtest.shape)
xtrain = xtrain.reshape(xtrain.shape[0], 50, 7, 1, 1)
xtest = xtest.reshape(xtest.shape[0], 50, 7, 1, 1)
ytrain, ytest = to_categorical(y_train, 3), to_categorical(y_test, 3)


#3D-CNN Tensorflow Model 
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(BatchNormalization(center=True, scale=True))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
model.summary()
# Fit data to model
history = model.fit(X_train, targets_train,batch_size=128,epochs=40,verbose=1,validation_split=0.3)
