import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
import os

import InputManager as inputManager
import CNNModel as Model

def getAcc(model, batch_size, dataset):
    torch.manual_seed(0)
    correct = 0
    total = 0
    data_loader = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)

    for data, label in data_loader:
        out = model(data)           # Compute prediction

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

        sig = nn.Sigmoid()
        out = sig(out).squeeze(1)   # Sigmoid goes in before computing loss i guess...

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

def train(model, train_dataset, val_dataset, lr, batch_size, num_epoch):
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
            optimizer.zero_grad()                           # Clean the previous step
            out = model(img)                               # Make prediction with model

            sig = nn.Sigmoid()
            out = sig(out).squeeze(1) # ce doesnt have sigmoid after it

            loss = criterion(out, label)            # Compute total losses
            loss.backward()                                 # Compute parameter updates
            optimizer.step()                                # Make the updates for each parameters
            #print(loss)
            tot_loss += loss.item()

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

Balanced_all_dataset, train_dataset, val_dataset, test_dataset, overfit_dataset = inputManager.getDataLoader()
model = Model.ECNN()

train(model, train_dataset, val_dataset, lr = 0.001, batch_size = 1000, num_epoch=30)
