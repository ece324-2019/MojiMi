from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import torch.nn.functional as F
import os

import InputManager as inputManager
import CNNModel as Model
import Models as Model_baseline

def getAcc(model, batch_size, dataset):
    torch.manual_seed(0)
    correct = 0
    total = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for data, label in data_loader:
        out = model(data)           # Compute prediction
        sig = nn.Sigmoid()          # Added
        out = sig(out).squeeze(1)   # Added

        # Compute accuracy before changing label to one hot encoding to use MSELoss
        pred = out.max(1)[1]        # Get the max prediction of all 10 probabilities and get the corresponding index
        correct += pred.eq(label.view_as(pred)).sum().item()
        total += data.shape[0]

    # Calculate actual accuracy and loss value
    accuracy = correct/total
    return accuracy

def evaluate(model, batch_size, dataset, criterion):
    torch.manual_seed(0)
    total_loss, correct, total = 0.0, 0, 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for i, (img, label) in enumerate(data_loader):
        out = model(img)           # Compute prediction

        # Compute loss
        total_loss += criterion(out, label).item()
        # loss = criterion(out, label)
        # total_loss += loss.item()

        # Compute accuracy before changing label to one hot encoding to use MSELoss
        pred = out.max(1)[1]        # Get the max prediction of all 10 probabilities and get the corresponding index
        correct += pred.eq(label.view_as(pred)).sum().item()
        total += label.shape[0]

    # Calculate actual accuracy and loss value
    accuracy = correct/total
    loss = total_loss/(i+1)
    return accuracy, loss

def train(model, train_dataset, val_dataset, lr, batch_size, num_epoch, save):
    torch.manual_seed(0)
    #optimizer = optim.SGD(model.parameters(), lr=lr)
    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True)
    iters, val_losses, train_losses, train_accs, val_accs = [], [], [], [], []

    start_time = time.time()
    for epoch in range(num_epoch):
        print('epoch:', epoch)
        tot_loss = 0
        for i, (img, label) in enumerate(train_data_loader):
            #print(i)
            optimizer.zero_grad()                           # Clean the previous step
            out = model(img)                               # Make prediction with model

            loss = criterion(out, label)            # Compute total losses
            loss.backward()                                 # Compute parameter updates
            optimizer.step()                                # Make the updates for each parameters
            tot_loss += loss.item()
            if i % 50 == 0:
                print('batch:', i)

        iters.append(epoch)
        train_losses.append(float(tot_loss)/(i+1))       # Computing average loss
        train_acc = getAcc(model, batch_size, train_dataset)
        train_accs.append(train_acc)
        print('Train_acc:', train_acc)
        # Prepare for plotting

        val_acc, val_loss = evaluate(model, batch_size, val_dataset, criterion)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        print('Val_acc:', val_acc)

    #torch.save(model.state_dict(), os.path.join(os.getcwd(), 'baseline_1.pt'))
    torch.save(model.state_dict(), os.path.join(os.getcwd(), save))
    end_time = time.time()
    time_used = end_time - start_time
    print('time:', time_used)

    # Plotting
    plt.figure(1)
    plt.title("Train vs Validation Loss")
    plt.plot(iters, train_losses, label='Train')
    plt.plot(iters, val_losses, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='best')

    plt.figure(2)
    plt.title("Train vs Validation Accuracy")
    plt.plot(iters, train_accs, label='Train')
    plt.plot(iters, val_accs, label='Val')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc='best')
    plt.show()
    return
import InputManager_5cls as inputManager_5cls
Balanced_all_dataset, train_dataset, val_dataset, test_dataset, overfit_dataset = inputManager_5cls.getDataLoader()
# Load pretrain model and set to not training
#model = models.resnet34(pretrained=True)
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
    #print(param)

''' # this is for resnet
num_ftrs = model.fc.in_features
#print(num_ftrs)
model.fc = nn.Linear(num_ftrs, 5)
'''

num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
                      nn.Linear(num_ftrs, 5),
                      )
# Find total parameters and trainable parameters
total_params = sum(p.numel() for p in model.parameters())
print(f'{total_params:,} total parameters.')
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f'{total_trainable_params:,} training parameters.')
# This method takes rediculously long
#train(model, overfit_dataset, overfit_dataset, lr = 0.001, batch_size = 64, num_epoch= 25, save = 'vgg16_overfit_0.pt')
#train(model, train_dataset, val_dataset, lr = 0.001, batch_size = 64, num_epoch= 2, save = 'ECNN_train_0.pt')

# Trying out Keras
from keras.applications.vgg16 import VGG16
pre_model = VGG16(weights = 'imagenet', include_top = False)

model = Model.keras_tl()

#overfit_dataset = inputManager_5cls.get_Input_Dataset_keras(pre_model, overfit_dataset)
#torch.save(overfit_dataset, os.path.join(os.getcwd(), 'cropped_pics_overfit_dataset_keras.pt'))
overfit_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_overfit_dataset_keras.pt'))

#train(model, overfit_dataset, overfit_dataset, lr = 0.0001, batch_size = 10, num_epoch=50, save ='keras_overfit.pt')

#train_dataset = inputManager_5cls.get_Input_Dataset_keras(pre_model, train_dataset)
#torch.save(train_dataset, os.path.join(os.getcwd(), 'cropped_pics_train_dataset_keras.pt'))
train_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_train_dataset_keras.pt'))

#val_dataset = inputManager_5cls.get_Input_Dataset_keras(pre_model, val_dataset)
#torch.save(val_dataset, os.path.join(os.getcwd(), 'cropped_pics_val_dataset_keras.pt'))
val_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_val_dataset_keras.pt'))

#test_dataset = inputManager_5cls.get_Input_Dataset_keras(pre_model, test_dataset)
#torch.save(test_dataset, os.path.join(os.getcwd(), 'cropped_pics_test_dataset_keras.pt'))
test_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_test_dataset_keras.pt'))

train(model, train_dataset, val_dataset, lr = 0.0001, batch_size = 300, num_epoch=100, save ='keras_train.pt')


#overfit_data_loader = torch.utils.data.DataLoader(overfit_dataset, batch_size=4, shuffle=True, num_workers=2)
#train(model, overfit_dataset, overfit_dataset, lr = 0.001, batch_size = 10, num_epoch=60, save = 'cropped_ECNN_overfit_5cls_64_1.pt')
#train(model, train_dataset, val_dataset, lr = 0.001, batch_size = 300, num_epoch=40, save = 'ECNN_train_5cls_64_1.pt')

#ecc_pt1 is the parameters for lr = 0.005, batch = 1000, num of epoch = 40
#train(model, overfit_dataset, overfit_dataset, lr = 0.001, batch_size = 64, num_epoch= 25, save = 'vgg16_overfit_0.pt')
#train(model, train_dataset, val_dataset, lr = 0.001, batch_size = 64, num_epoch= 2, save = 'ECNN_train_0.pt')

#model_1 = Model_baseline.Baseline_64()
#train(model_1, train_dataset, val_dataset, lr = 0.001, batch_size = 9000, num_epoch= 70, save ='baseline_3.pt')


