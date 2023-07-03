#%% Image Tensor dataframe
import zipfile
import nibabel
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import gzip
import shutil
import tempfile
import os
import time
import timeit
import numpy as np
import torch

#%%
patient_id = []
filelist = []
tensors = []

for zipfile_path in list(Path('projT1/20252').glob('*.zip')):   

    with zipfile.ZipFile(zipfile_path) as zf:
        if 'T1/T1.nii.gz' in zf.namelist():
            with zf.open('T1/T1.nii.gz') as zip_file:
                with tempfile.NamedTemporaryFile(delete = False, suffix = '.nii.gz') as temp_file:
                    temp_file.write(zip_file.read())
                    temp_path = temp_file.name
                #tensor_data = nibabel.load(temp_path).get_fdata()
                #tensors.append(torch.from_numpy(tensor_data))
                os.remove(temp_path)
                patient_id.append(zipfile_path.stem)
        else: 
            #tensors.append(None)
            patient_id.append(np.nan)
        
df_image = pd.DataFrame({'eid': patient_id})
df_image = df_image.dropna()

#%%
import sqlite3
#Creating age and gender dataframes
con = sqlite3.connect("ukbiobank.db")

df1 = pd.read_sql("select * from '20252';", con)
df1.drop(columns=['instance_index', 'array_index'], inplace = True)
print(df1.head())

#df2 = pd.read_sql("select * from '34';", con)
#df2.drop(columns=['instance_index', 'array_index'], inplace= True)
#df2.rename(columns={"34": "Year of Birth"}, inplace=True)

#df3 = pd.read_sql("select * from '53';", con)
#df3.rename(columns={"53": "Date of Visit"}, inplace=True)
#df3['Date of Visit'] = pd.to_datetime(df3['Date of Visit'], errors='coerce')
#df3['Year of Visit'] = df3['Date of Visit'].dt.year
#df3.drop(columns=['instance_index', 'array_index', 'Date of Visit'], inplace= True)

df4 = pd.read_sql("select * from '20500';", con)
df4.drop(columns=['instance_index', 'array_index'], inplace= True)
df4.rename(columns={"20500": "mental illness"}, inplace=True)
print(df4.head())

df_merged = df1.merge(df4, on = 'eid', how = 'inner')
#df_merged["Age"] = df_merged['Year of Visit'] - df_merged['Year of Birth']
#df_merged.loc[(df_merged.Age < 55),  'AgeGroup'] = '0'
#df_merged.loc[(df_merged.Age > 65),  'AgeGroup'] = '1'
#df_age = df_merged.drop(columns=['Year of Birth', 'Year of Visit', 'Gender', 'Age'])
print(df_merged.head())

#df_age = df_merged.drop(columns=['Year of Birth', 'Year of Visit', 'Age', 'AgeGroup'])
df_gender = df_merged.dropna()



df_gender['eid'] = df_gender['eid'].astype(str)
df_gender['20252'] = df_gender['20252'].astype(str)
df_gender['eid'] = df_gender['eid'].str.cat(df_gender['20252'], sep='_').str.replace(' ', '_')
df_gender.drop(columns=['20252'], inplace= True)
print(df_gender.head())
print(df_gender.shape)
#sns.kdeplot(df['Age'])
## sns.distplot(df['Age'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth':3}, label = Age)
#plt.show()  
df_gender = df_gender[df_gender['mental illness'] != -121]
df_gender = df_gender[df_gender['mental illness'] != -818]
value_to_drop = 0
fraction_to_drop = 0.5

df_dropped = df_gender.groupby('mental illness').apply(lambda x: x.sample(frac=fraction_to_drop) if x.name == value_to_drop else x).reset_index(drop=True)

gender_dist = df_dropped['mental illness'].value_counts()

print(gender_dist)
gender_dist.plot.pie()
con.close()

# Merge df_image and df_age
df_final = df_image.merge(df_dropped, on='eid', how='left')

# Drop duplicates and NaN values
duplicates_count = df_final['eid'].duplicated().sum()
df_final.drop_duplicates(subset=['eid'], inplace=True)
df_final = df_final.dropna()

con.close()

print(df_final.head())
# %%
"""# final dataframe
#df_age['eid'] =  df_age['eid'].astype(str)
#df_image['eid'] = df_image['eid'].astype(str)

df_final = df_image.merge(df_gender, on = 'eid', how = 'left')
print(df_final.shape)
duplicates_count = df_final['eid'].duplicated().sum()
print(duplicates_count)
df_final.drop_duplicates(subset=['eid'], inplace = True)
df_final = df_final.dropna()
print(df_final.shape)
print(df_final)"""
# %%
#Creating the train and test csv files

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_final, test_size=0.1)

print(train_df.shape)
train_df = train_df
print(train_df.shape)
train_df.to_csv('BrainPrediction/train_mental.csv', index = False)
test_df.to_csv('BrainPrediction/test_mental.csv', index = False)
# %%

