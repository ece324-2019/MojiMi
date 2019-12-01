from torchvision import datasets, models, transforms

import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os

import InputManager as inputManager_5cls
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

Balanced_all_dataset, train_dataset, val_dataset, test_dataset, overfit_dataset = inputManager_5cls.getDataLoader()

# Transfer learninf for training VGG16 from Pytorch
# Load pretrain model and set to not training
model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

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




# Try Pytorch Inputting the 1000 classification as input instead of modifying last layer
# Load pretrain model and set to not training
pre_model = models.vgg16(pretrained=True)
for param in pre_model.parameters():
    param.requires_grad = False

num_ftrs = pre_model.classifier[6].out_features # 1000
print(num_ftrs)
model = Model.ENN_0_5cls(num_ftrs)

#overfit_dataset = inputManager_5cls.get_Input_Dataset(pre_model, overfit_dataset)
#torch.save(overfit_dataset, os.path.join(os.getcwd(), 'cropped_pics_overfit_dataset.pt'))
overfit_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_overfit_dataset.pt'))

#train_dataset = inputManager_5cls.get_Input_Dataset(pre_model, train_dataset)
#torch.save(train_dataset, os.path.join(os.getcwd(), 'cropped_pics_train_dataset.pt'))
train_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_train_dataset.pt'))

#val_dataset = inputManager_5cls.get_Input_Dataset(pre_model, val_dataset)
#torch.save(val_dataset, os.path.join(os.getcwd(), 'cropped_pics_val_dataset.pt'))
##val_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_val_dataset.pt'))

#test_dataset = inputManager_5cls.get_Input_Dataset(pre_model, test_dataset)
#torch.save(test_dataset, os.path.join(os.getcwd(), 'cropped_pics_test_dataset.pt'))
test_dataset = torch.load(os.path.join(os.getcwd(), 'cropped_pics_test_dataset.pt'))

#overfit_data_loader = torch.utils.data.DataLoader(overfit_dataset, batch_size=4, shuffle=True, num_workers=2)
#train(model, overfit_dataset, overfit_dataset, lr = 0.001, batch_size = 10, num_epoch=60, save = 'cropped_ECNN_overfit_5cls_64_1.pt')
#train(model, train_dataset, val_dataset, lr = 0.001, batch_size = 300, num_epoch=40, save = 'ECNN_train_5cls_64_1.pt')


model_final = Model.ECNN_final()
model_final.load_state_dict(torch.load('ECNN.pt'))
model_final.eval()
'''
model_ECNN.eval()
acc = getAcc(model_ECNN, batch_size, test_dataset) # dataset will be the testset
'''

# Plotting Confusion Matrix
from sklearn.metrics import confusion_matrix
#val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True)
test_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(test_dataset), shuffle=True)
for data, label in test_data_loader:
    out = model_final(data)
    pred = out.max(1)[1]  # Get the max prediction of all 10 probabilities and get the corresponding index
    print(label)
    label_int = label
    #label_int = label.max(1)[1]
# angry, hapoy, neurtal, sad, surprise
print(confusion_matrix(pred, label_int, labels=[0,1,2,3,4]))


# Keras VGG16
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