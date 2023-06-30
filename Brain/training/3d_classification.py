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
# 5. Parameters
batch_size = 16
num_epochs = 10
n_outputs = 1
num_classes = 1
num_channels = 1
csv_file  = 'BrainPrediction/train_gender.csv'
threshold = 0
#%% 
# 6. Define your custom dataset

class BrainDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, tio, deterministic=True):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.tio = tio.Compose([
            tio.CropOrPad((128,128,32)),   # Resize images to a consistent size 
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
train_dataset = BrainDataset(csv_file= 'BrainPrediction/train_gender.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", tio=tio)
val_dataset = BrainDataset(csv_file = 'BrainPrediction/test_gender.csv', root_dir="/mnt/bulk/radhika/project/projT1/20252", tio=tio)  # Assuming 'test.csv' as the validation dataset

# Define the data loader for training
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers=8,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle=False, num_workers=8,drop_last=True)


#%% 
# 8. Define the network

import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_channels, num_classes):
        super(Net, self).__init__()
        self.conv1 = nn.Conv3d(num_channels, 16, kernel_size=3, stride=1, padding=(0,0,1))
        self.max1 = nn.MaxPool3d(kernel_size = 2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=(0,0,1))
        self.max2 = nn.MaxPool3d(kernel_size = 2)
        self.conv3 = nn.Conv3d(32, 32, kernel_size=5, stride=1, padding=(0,0,2))
        self.max3 = nn.MaxPool3d(kernel_size = 2)
        self.conv4 = nn.Conv3d(32, 32, kernel_size=5, stride=1, padding=(0,0,2))
        self.max4 = nn.MaxPool3d(kernel_size = 2)
        self.fc = nn.Linear(1024, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        #print(x.shape)
        x = self.max1(self.relu(self.conv1(x)))
        #print(x.shape)
        x = self.max2(self.relu(self.conv2(x)))
        #print(x.shape)
        x = self.max3(self.relu(self.conv3(x)))
        #print(x.shape)
        x = self.max4(self.relu(self.conv4(x)))
        #print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    
class GenderClassifier(nn.Module):
    def __init__(self, input_dim):
        super(GenderClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x



#Create an instance of your model
model = Net(num_channels = num_channels, num_classes = num_classes).cuda()  #to move the model from CPU to device memory that is gpu(cuda) memory

#%% 
# 9. Define loss and optimizer

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#%% 
# 10. Running the loops 

train_losses = []
val_losses = []
train_outputs = []
train_labels = []
val_outputs = []
val_labels = []


# Initialize the running loss for this epoch

for epoch in range(num_epochs):

    # Training loop
    
    # Set the model to train mode
    model.train()

    running_train_loss = 0.0

    # Iterate over the training dataset
    for i, (images, labels) in enumerate(train_loader):

        images = images.squeeze(0).cuda()
        #print(images.shape)
        labels = labels.float().cuda()
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images).squeeze(1)
        # Append the outputs and labels to the respective lists
        
        train_outputs.extend(outputs.tolist())
        #train_outputs.extend(( outputs > threshold ).float().tolist())
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
    train_loss = running_train_loss / len(train_loader)
    train_losses.append(train_loss)

        
    # Validation loop
    model.eval()  # Set the model to evaluation mode

    # Initialize the running loss and accuracy for validation    

    running_val_loss = 0.0
    with torch.no_grad():
        for j, (images, labels) in enumerate(val_loader):
            
            images = images.squeeze(0).cuda()
            #print(images.shape)
            labels = labels.float().cuda()
            # Forward pass
        
            outputs = model(images).squeeze(1)
            # Append the outputs and labels to the respective lists
            val_outputs.extend(outputs.tolist())
            #val_outputs.extend(( outputs > threshold ).float().tolist())
            val_labels.extend(labels.tolist())


            # Calculate the loss
            loss = criterion(outputs, labels)

            # Update the running loss
            running_val_loss += loss.item() 

    # Calculate the average validation loss
    val_loss = running_val_loss / len(val_loader)
    val_losses.append(val_loss)

    # Print the validation loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')   
    
#%% 

#outputs = ( outputs > threshold ).float()

#%%

# 11. After training, you can save the model
torch.save(model.state_dict(), "trained_model.pth")
# %%
from BrainPrediction import visualizations

#visualizations.scatter_plot(labels, outputs)
visualizations.loss_curves(train_losses, val_losses)
#visualizations.distribution_plots(outputs, labels)



#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

from sklearn.metrics import confusion_matrix, auc
import numpy as np
cm_train = confusion_matrix(train_labels, (val_outputs > threshold ).float().tolist())
print(f'Train matrix: \n {cm_train}')
cm_val = confusion_matrix(val_labels, (val_outputs > threshold ).float().tolist())
print(f'Val Matrix: \n {cm_val}')


# %%
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(val_labels, val_outputs)
roc_auc = auc(fpr, tpr) 
# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
from sklearn.metrics import roc_curve, auc
fpr, tpr, thresholds = roc_curve(train_labels, train_outputs)
roc_auc = auc(fpr, tpr) 
# Plot the ROC curve
plt.plot(fpr, tpr, color='blue', label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
# %%
