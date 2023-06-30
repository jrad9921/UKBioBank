#%% 
# 0. Imports 
import zipfile
import nibabel
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import os
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import pandas as pd
import torchio as tio
from torchvision.transforms import Resize, ToTensor
import dateutil
dateutil.__version__
import torch.nn.functional as F


#%%
# 1. Image Tensor dataframe
patient_id = []
filelist = []
tensors = []

for zipfile_path in list(Path('projT1/20252').glob('*.zip'))[:100]:   

    with zipfile.ZipFile(zipfile_path) as zf:
        if 'T1/T1.nii.gz' in zf.namelist():
            with zf.open('T1/T1.nii.gz') as zip_file:
                with tempfile.NamedTemporaryFile(delete = False, suffix = '.nii.gz') as temp_file:
                    temp_file.write(zip_file.read())
                    temp_path = temp_file.name
                #tensor_data = nibabel.load(temp_path).get_fdata()[:,:,:]
                #tensors.append(torch.from_numpy(tensor_data))
                os.remove(temp_path)
                patient_id.append(zipfile_path.stem)
        else: 
            tensors.append(None)
            patient_id.append(np.nan)
        
df_image = pd.DataFrame({'eid': patient_id})
df_image = df_image.dropna()

print(df_image)
print(df_image.shape)

# %% 
# 2. Age dataframe 
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
# 3. Final dataframe

df_final = df_image.merge(df_age, on = 'eid', how = 'left')
print(df_final.shape)
duplicates_count = df_final['eid'].duplicated().sum()
print(duplicates_count)
df_final.drop_duplicates(subset=['eid'], inplace = True)
df_final = df_final.dropna()
print(df_final.shape)
print(df_final)
# %%
# 4. Creating the train and test csv files

from sklearn.model_selection import train_test_split
train_df, test_df = train_test_split(df_final, test_size=0.1)

print(train_df.shape)
train_df = train_df
print(train_df.shape)
train_df.to_csv('BrainPrediction/train_age.csv', index = False)
test_df.to_csv('BrainPrediction/test_age.csv', index = False)
#%%


#%%
# 5. Parameters
batch_size = 44
num_epochs = 5
n_outputs = 1
num_classes = 1
num_channels = 1
csv_file  = 'BrainPrediction/train_age.csv'

#%% 
# 6. Define your custom dataset

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, tio, deterministic=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tio = tio.Compose([
            tio.CropOrPad((128, 128, 128)),   # Resize images to a consistent size 
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
                img_data = nibabel.load(temp_path).get_fdata()
                os.remove(temp_path)
                patient_id.append(zipfile_path.stem)
        image_tensor = torch.tensor(img_data.astype(np.float32))
        image_tensor = image_tensor.unsqueeze(0)

        labels = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.tio:
            images = self.tio(image_tensor)

        # Ensure the image tensor has the correct shape
        #images = images.unsqueeze(0)  # Add a batch dimension
        # Return the image and label
        return images, labels

"""
        # Create a new tensor with the desired size and copy the values
        resized_image = torch.empty((1, self.resize_size, self.resize_size))
        resized_image.copy_(F.interpolate(image_tensor, size=(self.resize_size, self.resize_size), mode='bilinear', align_corners=False))

        # Remove the batch dimension
        image = resized_image.squeeze(0)
"""


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
# 7. Create Dataloaders

#Create Datasets for training and testing
train_dataset = BrainDataset(csv_file= 'BrainPrediction/train_age.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", tio=tio)
val_dataset = BrainDataset(csv_file = 'BrainPrediction/test_age.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", tio=tio)  # Assuming 'test.csv' as the validation dataset

# Define the data loader for training
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=8)


#%% 
# 8. Define the network

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_channels, num_outputs):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(16, 16, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(1 * 16 * 128 * 128 * 128, num_outputs)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


#Create an instance of your model
model = Net(num_channels = num_channels, num_classes = num_classes).cuda()  #to move the model from CPU to device memory that is gpu(cuda) memory

#%% 
# 9. Define loss and optimizer

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

#%% 
# 10. Running the loops 

train_losses = []
val_losses = []
train_outputs = []
train_labels = []
val_outputs = []
val_labels = []

for epoch in range(num_epochs):

    # Training loop
    
    # Set the model to train mode
    model.train()

    # Initialize the running loss for this epoch
    running_train_loss = 0.0

    # Iterate over the training dataset
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.squeeze(0).cuda()
        labels = labels.cuda().float()

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Append the outputs and labels to the respective lists
        train_outputs.extend(outputs.tolist())
        train_labels.extend(labels.tolist())

        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the running loss
        running_train_loss += loss.item()

    # Calculate the average training loss for this epoch
    train_loss = running_train_loss / len(train_dataset)
    train_losses.append(train_loss)

        
    # Validation loop
    model.eval()  # Set the model to evaluation mode

    # Initialize the running loss and accuracy for validation    
    running_val_loss = 0.0

    for j, (images, labels) in enumerate(val_loader):

        images = images.squeeze(0).cuda()
        labels = labels.cuda().float()

        # Forward pass
        with torch.no_grad():
            outputs = model(images)

        # Append the outputs and labels to the respective lists
        val_outputs.extend(outputs.tolist())
        val_labels.extend(labels.tolist())


        # Calculate the loss
        loss = criterion(outputs, labels.float().cuda())

        # Update the running loss
        running_val_loss += loss.item() 

    # Calculate the average validation loss
    val_loss = running_val_loss / len(val_dataset)
    val_losses.append(val_loss)

    # Print the validation loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {torch.tensor(train_losses).mean():.4f}, Val Loss: {torch.tensor(val_losses).mean():.4f}')   
    
#%% 
# 11. After training, you can save the model
#torch.save(model.state_dict(), "trained_model.pth")
# %%
# 12. Plot the Loss Curves
plt.figure(figsize=[8,4])
plt.plot(train_losses,'r', linewidth=2.0)
plt.plot(val_losses,'b', linewidth=2.0)
plt.legend(['Training loss', 'Validation loss'], fontsize=10)
plt.xlabel('Epochs ', fontsize=10)
plt.ylabel('Loss', fontsize=10)
plt.title('Loss Curves', fontsize=10)
plt.show()
#%%
# 13. Scatter plot
labels = labels.cpu()
outputs= outputs.cpu()
plt.figure(figsize= [4,4])
plt.scatter(train_labels, train_outputs, s = 5)
plt.xlabel('True ages')
plt.ylabel('Predicted ages')
plt.title('Scatter Plot: Predicted vs. True ages')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()
#%%
# 14. Age distribution plot
labels = labels
outputs = outputs
plt.figure(figsize = [4,4])
sns.distplot(labels, label='True ages')
sns.distplot(outputs.detach().numpy(), label='Predicted ages')
plt.xlabel('Age')
plt.ylabel('Density')
plt.title('Distribution Plot')
plt.legend()
plt.show()
# %%
