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

for zipfile_path in list(Path('20252').glob('*.zip'))[:1000]:   

    with zipfile.ZipFile(zipfile_path) as zf:
        if 'T1/T1.nii.gz' in zf.namelist():
            with zf.open('T1/T1.nii.gz') as zip_file:
                with tempfile.NamedTemporaryFile(delete = False, suffix = '.nii.gz') as temp_file:
                    temp_file.write(zip_file.read())
                    temp_path = temp_file.name
                #tensor_data = nibabel.load(temp_path).get_fdata()[100,:,:]
                #tensors.append(torch.from_numpy(tensor_data))
                patient_id.append(zipfile_path.stem)
                os.remove(temp_path)
        else: 
            tensors.append(None)
            patient_id.append(np.nan)
        
df_image = pd.DataFrame({'eid': patient_id})
df_image = df_image.dropna()

print(df_image)
print(df_image.shape)

# %% Age dataframe 
import sqlite3
import seaborn as sns
#Creating age and gender dataframes
con = sqlite3.connect("ukbiobank.db")

df1 = pd.read_sql("select * from '20252';", con)
#duplicates_count  = df1['eid'].duplicated().sum()
#print(df1['eid'].duplicated())
#print(duplicates_count)
#df_sp = df1[df1['eid'] == 1000010] 
df1.drop(columns=['instance_index', 'array_index'], inplace = True)
print(df1.head())

df2 = pd.read_sql("select * from '34';", con)
df2.drop(columns=['instance_index', 'array_index'], inplace= True)
df2.rename(columns={"34": "Year of Birth"}, inplace=True)
print(df2.head())

df3 = pd.read_sql("select * from '53';", con)
df3.rename(columns={"53": "Date of Visit"}, inplace=True)
df3['Date of Visit'] = pd.to_datetime(df3['Date of Visit'], errors='coerce')
df3['Year of Visit'] = df3['Date of Visit'].dt.year
df3.drop(columns=['instance_index', 'array_index', 'Date of Visit'], inplace= True)
print(df3.head())

#df4 = pd.read_sql("select * from '31';", con)
#df4.drop(columns=['instance_index', 'array_index'], inplace= True)
#df4.rename(columns={"31": "Gender"}, inplace=True)
#print(df4.head())

df_merged = df1.merge(df2, on = 'eid', how = 'inner').merge(df3, on = 'eid', how = 'inner')
df_merged["Age"] = df_merged['Year of Visit'] - df_merged['Year of Birth']
df_age = df_merged.drop(columns=['Year of Birth', 'Year of Visit'])

print(df_age.head())
sns.kdeplot(df_age['Age'])
df_age['eid'] = df_age['eid'].dropna()
#df_merged.loc[(df_merged.Age < 55),  'AgeGroup'] = '0'
#df_merged.loc[(df_merged.Age > 65),  'AgeGroup'] = '1'

#df_gender = df.drop(columns=['Year of Birth', 'Year of Visit', 'Age', 'AgeGroup'])
#print(df_gender.head())
#print(df_gender.shape)


## sns.distplot(df['Age'], hist = False, kde = True, kde_kws = {'shade': True, 'linewidth':3}, label = Age)
#plt.show()  
df_age['eid'] = df_age['eid'].astype(str)
df_age['20252'] = df_age['20252'].astype(str)
df_age['eid'] = df_age['eid'].str.cat(df_age['20252'], sep='_').str.replace(' ', '_')
df_age.drop(columns=['20252'], inplace=True)

# Merge df_image and df_age
df_final = df_image.merge(df_age, on='eid', how='left')

# Drop duplicates and NaN values
duplicates_count = df_final['eid'].duplicated().sum()
df_final.drop_duplicates(subset=['eid'], inplace=True)
df_final = df_final.dropna()
con.close()

print(df_age.head())
# %%
# final dataframe
#df_age['eid'] =  df_age['eid'].astype(str)
#df_image['eid'] = df_image['eid'].astype(str)

df_final = df_image.merge(df_age, on = 'eid', how = 'left')
print(df_final.shape)
duplicates_count = df_final['eid'].duplicated().sum()
print(duplicates_count)
df_final.drop_duplicates(subset=['eid'], inplace = True)
df_final = df_final.dropna()
print(df_final.shape)
print(df_final)
# %%
#Creating the train and test csv files

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_final, test_size=0.1)

print(train_df.shape)
train_df = train_df
print(train_df.shape)
train_df.to_csv('BrainPrediction/train_age.csv', index = False)
test_df.to_csv('BrainPrediction/test_age.csv', index = False)
#%%
# 1. imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
import ipympl
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import pandas as pd

from torchvision.transforms import Resize, ToTensor
import dateutil
dateutil.__version__
import torch.nn.functional as F

#%%
# 2. Parameters
batch_size = 100
num_epochs = 20
n_outputs = 1
n_classes = 2
csv_file  = 'BrainPrediction/train_gender.csv'
#%% 
# 3. Define your custom dataset

import ast
import re
import zipfile
import tempfile

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform, deterministic=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),   # Resize images to a consistent size
            transforms.CenterCrop(256),  
            transforms.ToTensor()  # Convert images to tensors
        ])
        self.deterministic = deterministic

    def __len__(self):
        # Return the size of the dataset
        return len(self.annotations)

    def __getitem__(self, index):
        eid = self.annotations.iloc[index, 0]
        #full_path = Path(self.root_dir)/self.annotations.iloc[index, 1]
        zipfilepath = os.path.join(self.root_dir, f"{eid}.zip")
        niftipath = 'T1/T1.nii.gz'

        with zipfile.ZipFile(zipfilepath) as zf:
            with zf.open(niftipath) as zip_file:
                with tempfile.NamedTemporaryFile(delete = False, suffix = '.nii.gz') as temp_file:
                    temp_file.write(zip_file.read())
                    temp_path = temp_file.name
                img_data = nibabel.load(temp_path).get_fdata()[100,:,:]
                os.remove(temp_path)
        image_tensor = torch.tensor(img_data.astype(np.float32))

        labels = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            images = self.transform(image_tensor)

        return(images, labels)
    
        # Ensure the image tensor has the correct shape
        image_tensor = image_tensor.unsqueeze(0)  # Add a batch dimension

        # Create a new tensor with the desired size and copy the values
        resized_image = torch.empty((1, self.resize_size, self.resize_size))
        resized_image.copy_(F.interpolate(image_tensor, size=(self.resize_size, self.resize_size), mode='bilinear', align_corners=False))

        # Remove the batch dimension
        image = resized_image.squeeze(0)

        # Return the image and label
        return image, label
"""        # Retrieve the tensor string from the CSV
        tensor_str = self.data.iloc[idx]['Tensors']

        # Convert the tensor string to a NumPy array
        tensor_np = np.array(ast.literal_eval(tensor_str))

        # Convert the NumPy array to a PyTorch tensor
        tensor = torch.from_numpy(tensor_np)

        # Apply resize transformation to the tensor
        tensor = tensor.reshape(self.resize_size)

        # Retrieve the label from the CSV
        label = self.data.iloc[idx]['Age']

        return tensor, label"""
    


#%% 
# 4. Create Dataloaders


#Create Datasets for training and testing
train_dataset = BrainDataset(csv_file = 'BrainPrediction/train_gender.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", transform=transforms)
val_dataset = BrainDataset(csv_file = 'BrainPrediction/test_gender.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", transform= transforms)  # Assuming 'test.csv' as the validation dataset

# Define the data loader for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#%% 
# 5. Define the network

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(1*256*256, 10) 
    self.l2 = nn.Linear(10, 10)
    self.l3 = nn.Linear(10, n_classes)
    self.do = nn.Dropout(0.2)  #if we're overfitting

  def forward(self, x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    do = self.do(h2 + h1)  #adding h1 here is the only thing that makes it different from the simple model
    logits = self.l3(do)
    return logits

#Create an instance of your model
model = Net().cuda()  #to move the model from CPU to device memory that is gpu(cuda) memory

#%% 
# 6. Define loss and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

#%% 
# 7. Running the loops 

train_losses = []
val_losses = []

# Training loop
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()

    # Initialize the running loss for this epoch
    running_loss = 0.0

    # Iterate over the training dataset
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.view(images.size(0), -1).cuda()
        labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float().cuda()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the running loss
        running_loss += loss.item() * batch_size

        # Calculate the average training loss for this epoch
        train_loss = running_loss / len(train_dataset)
    train_losses.append(train_loss)

     # Initialize the running loss and accuracy for validation    
    running_val_loss = 0.0

    for j, (images, labels) in enumerate(val_loader):  # for every epoch, get a batch from the training loader
            
        # x: b*1*256*256
        #b = images.size(0)
        images = images.view(images.size(0), -1).cuda()  # reshape, multiplies b with everything in x that is not b, here it is 1*256*256
        labels = torch.nn.functional.one_hot(torch.as_tensor(labels)).float().cuda()

        # 1. forward
        with torch.no_grad():
            outputs = model(images.cuda())  # l=logits

        # 2. compute the objective function
        loss = criterion(outputs, labels)  # comparison of predictions with ground truth

        running_val_loss += loss.item() * batch_size
    
        val_loss = running_val_loss /len(val_loader)
    val_losses.append(val_loss)
    
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {torch.tensor(train_losses).mean():.4f}, Validation Loss: {torch.tensor(val_losses).mean():.4f}')   


        #if (j+1) % 1 == 0:
            #print(f'Validation: epoch {epoch+1}/{nb_epochs}, step {j+1}/{len(val_loader)}, outputs {outputs.size(0)}, Val loss: {torch.tensor(val_losses).mean():.2f}')   
  
      
'''
    # Validation loop
    model.eval()  # Set the model to evaluation mode

    # Initialize the running loss and accuracy for validation
    val_loss = 0.0

    # Disable gradient computation
    with torch.no_grad():
        for tensors, labels in val_loader:
            # Forward pass
            outputs = model(tensors)

            # Calculate the loss
            loss = criterion(outputs, labels)

            # Update the running loss
            val_loss += loss.item() * tensors.size(0)

    # Calculate the average validation loss
    val_loss /= len(val_dataset)

    # Print the validation loss for this epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss}")

'''
#%% Visualizations
import BrainPrediction.visualizations as visualizations

visualizations.scatter_plot(outputs, labels)
visualizations.loss_curves(train_losses, val_losses)


# 8. After training, you can save the model
#torch.save(model.state_dict(), "trained_model.pth")
