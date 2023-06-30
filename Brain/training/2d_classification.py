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
#%%
# 2. Parameters
batch_size = 50
num_epochs = 10
n_outputs = 1
n_classes = 1
threshold = 0
csv_file  = 'BrainPrediction/train_gender.csv'
#%% 
# 3. Define your custom dataset


import zipfile
import tempfile

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform, deterministic=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform.Compose([
            transforms.ToPILImage(),
            transforms.Resize(200),   # Resize images to a consistent size
            transforms.CenterCrop(200),  
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
    


#%% 
# 4. Create Dataloaders


#Create Datasets for training and testing
train_dataset = BrainDataset(csv_file=csv_file, root_dir="/mnt/bulk/radhika/project/projT1/20252", transform=transforms)
val_dataset = BrainDataset(csv_file =csv_file, root_dir="/mnt/bulk/radhika/project/projT1/20252", transform= transforms)  # Assuming 'test.csv' as the validation dataset

# Define the data loader for training
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

#%% 
# 5. Define the network

class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(1*200*200, 10) 
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
train_labels = []
val_labels = []
train_outputs = []
val_outputs = []
# Training loop
for epoch in range(num_epochs):
    # Set the model to train mode
    model.train()

    # Initialize the running loss for this epoch
    running_train_loss = 0.0

    # Iterate over the training dataset
    for i, (images, labels) in enumerate(train_loader):
        
        images = images.view(images.size(0), -1).cuda()
        labels = labels.float().cuda()
        train_labels.extend(labels.cpu().numpy().tolist())
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).squeeze(1)
        train_outputs.extend(outputs.cpu().numpy().tolist())
        # Calculate the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        # Update the running loss
        running_train_loss += loss.item() 

    # Calculate the average training loss for this epoch
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    running_val_loss = 0.0        
        
    for j, (images, labels) in enumerate(val_loader):  # for every epoch, get a batch from the training loader
        

        # x: b*1*256*256
        #b = images.size(0)
        images = images.view(images.size(0), -1).cuda()  # reshape, multiplies b with everything in x that is not b, here it is 1*256*256
        labels = labels.float().cuda()
        val_labels.append(labels)
        # 1. forward
        with torch.no_grad():
            outputs = model(images).squeeze(1)  # l=logits
        val_outputs.append(outputs)
        # 2. compute the objective function
        loss = criterion(outputs, labels)  # comparison of predictions with ground truth

        running_val_loss += loss.item() 
    val_loss = running_val_loss /len(val_loader)
    val_losses.append(val_loss)
        
        #if (j+1) % 1 == 0:
            #print(f'Validation: epoch {epoch+1}/{nb_epochs}, step {j+1}/{len(val_loader)}, outputs {outputs.size(0)}, Val loss: {torch.tensor(val_losses).mean():.2f}')   
   
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {torch.tensor(train_losses).mean():.4f}, Val Loss: {torch.tensor(val_losses).mean():.4f}')   
      
#%%
outputs = ( outputs > threshold ).float()
#%% 

# 8. After training, you can save the model
torch.save(model.state_dict(), "trained_model.pth")
# %%
#torch.load('trained_model.pth')
import BrainPrediction.visualizations as visualizations
classes = ['0', '1']
visualizations.loss_curves(train_losses, val_losses)
#visualizations.confusion_matrix(train_labels, train_outputs, classes) 
#visualizations.scatter_plot(train_outputs, train_labels)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(labels.cpu(), outputs.cpu())
print(cm)
# %%
