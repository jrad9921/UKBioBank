#%%
# 1. imports
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from customDataset import BrainDataset
import matplotlib.pyplot as plt
import ipympl
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import pandas as pd

#%%
# 2. Parameters
nb_classes = 2
n_output = 1
learning_rate = 1e-2
batch_size = 10
nb_epochs = 20 # an epoch is a full pass through the dataset
train_split = 0.8
csv_file = 'train_age.csv'
n_iterations = 20

#%%
# 3. Loading images
dataset = BrainDataset(csv_file=csv_file, root_dir="/mnt/bulk/radhika/project/20219", transform=transforms)
train_size = int(train_split * len(dataset))
test_size = len(dataset) - train_size
train_set, val_set = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8)
# %%
# 4. Define the model
# 4.2. resNet model
class Net(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(1*256*256, 64) 
    self.l2 = nn.Linear(64, 64)
    self.l3 = nn.Linear(64, n_output)
    self.do = nn.Dropout(0.2)  #if we're overfitting

  def forward(self, x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    do = self.do(h2 + h1)  #adding h1 here is the only thing that makes it different from the simple model
    logits = self.l3(do)
    return logits

model = Net().cuda()  #to move the model from CPU to device memory that is gpu(cuda) memory
# %%
# 5. Define the loss and optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-2)  # params = model.parameters()
criterion = nn.MSELoss()

#%%
# 6. Training and validation
from tqdm import tqdm, trange
train_losses = []
#train_r2 = []
val_losses = []
#val_accuracies = list()
#r2_scores = []
#train_accuracies = list()    
    
for epoch in range(nb_epochs):
    for i, (images, labels) in enumerate(train_loader):  # for every epoch, get a batch from the training loader

            
        running_train_loss = 0.0  # image and the labels

        # x: b*1*256*256
        b = images.size(0)
        images = images.view(images.size(0), -1).cuda()  # reshape, multiplies b with everything in x that is not b, here it is 3*256*256
        labels = labels.unsqueeze(1)
        # 1. forward
        outputs = model(images)  # l=logits

        # 2. compute the objective function
        loss = criterion(outputs, labels.float().cuda())  # comparison of predictions with ground truth

        # 3. cleaning the gradients
        optimizer.zero_grad()  # params.grad._zero()

        # 4. accumulate partial derivatives of J wrt params
        loss.backward()  # params.grad.add_(dJ/dparams)

        # 5. step in opposite direction of the gradient
        optimizer.step()  # params = params - eta * params.grad

        running_train_loss += loss.item() * batch_size
        train_loss = running_train_loss /len(train_loader)
        train_losses.append(train_loss)
        #if (i+1) % 1 == 0:
            #print(f'Training: epoch {epoch+1}/{nb_epochs}, step {i+1}/{len(train_loader)}, outputs {outputs.size(0)}, Train loss: {torch.tensor(train_losses).mean():.2f}')   
        #r2 = r2_score(labels.float().cpu().detach().numpy(), outputs.cpu().detach().numpy())
        #train_losses.append(loss.item())
        #train_accuracies.append(y.eq(l.detach().argmax(dim=1).cpu()).float().mean())
        #train_r2.append(r2).mean()
    
    #print(f"Epoch {epoch +1}, Train loss: {torch.tensor(train_losses).mean():.2f}")
    #print(f"Training accuracy with DenseNet: {torch.tensor(train_accuracies).mean():.2f}")
    #print(f'Train r2 Score: {train_r2.mean():.2f}')

    #Validation loading
    for j, (images, labels) in enumerate(val_loader):  # for every epoch, get a batch from the training loader
        
        running_val_loss = 0.0        
        
        # x: b*1*256*256
        #b = images.size(0)
        images = images.view(images.size(0), -1)  # reshape, multiplies b with everything in x that is not b, here it is 1*256*256
        labels = labels.unsqueeze(1)

        # 1. forward
        with torch.no_grad():
            outputs = model(images.cuda())  # l=logits

        # 2. compute the objective function
        loss = criterion(outputs, labels.float().cuda())  # comparison of predictions with ground truth

        running_val_loss += loss.item() * batch_size
        val_loss = running_val_loss /len(val_loader)
        val_losses.append(val_loss)
        
        #if (j+1) % 1 == 0:
            #print(f'Validation: epoch {epoch+1}/{nb_epochs}, step {j+1}/{len(val_loader)}, outputs {outputs.size(0)}, Val loss: {torch.tensor(val_losses).mean():.2f}')   
   
        print(f'Epoch [{epoch+1}/{nb_epochs}], Train Loss: {torch.tensor(train_losses).mean():.4f}, Val Loss: {torch.tensor(val_losses).mean():.4f}')   
        
        #val_losses.append(loss.cpu().detach().item())
        #val_accuracies.append(y.eq(l.detach().argmax(dim=1).cpu()).float().mean())
        
        #r2 = r2_score(labels.float().cpu(), outputs.cpu())
        

   # if (epoch) % 1 == 0:    
      #r2 = r2_score(outputs.cpu(), labels.cpu())
      #r2_scores.append(r2)
      #print(f"Epoch {epoch +1}, Train loss: {torch.tensor(train_losses).mean():.2f}, Val loss: {torch.tensor(val_losses).mean():.2f}")

      
      #print(f"Val accuracy with DenseNet: {torch.tensor(val_accuracies).mean():.2f}")
      #print(f'Val r2 Score: {val_r2.mean():.2f}')  
#%%
# 7. Plot the Loss Curves
plt.figure(figsize=[8,4])
plt.plot(train_losses,'r',linewidth=2.0)
plt.plot(val_losses,'b',linewidth=2.0)
plt.legend(['Training loss', 'Validation Loss'],fontsize=10)
plt.xlabel('Epochs ',fontsize=10)
plt.ylabel('Loss',fontsize=10)
plt.title('Loss Curves',fontsize=10)
plt.show()
#%%
# 8. Scatter plot
labels = labels.cpu()
outputs= outputs.cpu()
plt.figure(figsize= [4,4])
plt.scatter(labels, outputs.detach().numpy(), s = 1)
plt.xlabel('True ages')
plt.ylabel('Predicted ages')
plt.title('Scatter Plot: Predicted vs. True ages')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.show()
#%%
# 9. Age distribution plot
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
#%%
'''
# Example predicted values, true labels, and genders
outputs = outputs.cpu()
labels = labels.cpu()
genders = pd.read_csv('train_gender.csv').values.tolist() 
# Define colors for each gender category
colors = {'1': 'blue', '0': 'red'}

# Scatter plot with different colors based on gender
plt.scatter(labels, outputs, c=[colors[gender] for gender in genders])

# Customize the plot
plt.xlabel('True Labels')
plt.ylabel('Predicted Values')
plt.title('Scatter Plot: Prediction vs. True Label')

# Add a legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Male', markerfacecolor='blue', markersize=8),
                   plt.Line2D([0], [0], marker='o', color='w', label='Female', markerfacecolor='red', markersize=8)]
plt.legend(handles=legend_elements)

# Display the plot
plt.show()
'''
'''
sns.kdeplot(df_merged['Age'])
#Plot the Accuracy Curves
plt.figure(figsize=[8,6])
plt.plot(torch.tensor(train_accuracies).mean(),'r',linewidth=3.0)
plt.plot(torch.tensor(val_accuracies).mean(),'b',linewidth=3.0)
plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
plt.xlabel('Epochs ',fontsize=16)
plt.ylabel('Accuracy',fontsize=16)
plt.title('Accuracy Curves',fontsize=16)   
plt.show()    
'''

# %%
