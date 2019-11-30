import os
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageOps
import skimage as sk
from skimage import util, transform
from distutils.dir_util import copy_tree
import random

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import Input_Dataset as input_Dataset

'''
In this file, it assumed all data will be stored in original file called cropped_pics
From calling getData(dataset_used), it obtains data from the chosen folder dataset_used. 
Function: plotPie() --> gives the distribution plot and number of images in each category

3 Data augmentation techniques are used: Horizontal flipping, adding random noise, adjust brightness and contrast
By calling dataAug('cropped_pics') will lead to a duplicate of cropped_pics named ori_cropped_pics for record, 
It will then augment the existed data inplace to 8x the number of images. 

Note, everytime, ori_cropped_pics will only be created if no previous ori_cropped_pics exist. 
So if dataset is renewed and want to augment again, delete ori_cropped_pics first then call: dataAug(dataset_used) 
'''
fileOrigin = os.getcwd()  # For running in local computer

# fileOrigin = os.path.join(os.getcwd(), 'gdrive/My Drive/Colab Notebooks/Mojimi') # For running on colab
#dataset_used = 'ori_cropped_pics'
dataset_used = 'cropped_pics_copy'

def getData(dataset_path):
    dataset_path = os.path.join(fileOrigin, dataset_path)

    angry_dataset = os.path.join(dataset_path, 'Angry')
    happy_dataset = os.path.join(dataset_path, 'Happy')
    neutral_dataset = os.path.join(dataset_path, 'Neutral')
    sad_dataset = os.path.join(dataset_path, 'Sad')
    surprised_dataset = os.path.join(dataset_path, 'Surprised')

    # Plot distribution
    angry_imgs = os.listdir(angry_dataset)
    happy_imgs = os.listdir(happy_dataset)
    print(happy_imgs)
    neutral_imgs = os.listdir(neutral_dataset)
    sad_imgs = os.listdir(sad_dataset)
    surprised_imgs = os.listdir(surprised_dataset)
    out = (angry_dataset, happy_dataset, neutral_dataset, sad_dataset, surprised_dataset,angry_imgs, happy_imgs, neutral_imgs, sad_imgs, surprised_imgs)
    return out

(angry_dataset, happy_dataset, neutral_dataset, sad_dataset, surprised_dataset, angry_imgs,happy_imgs, neutral_imgs, sad_imgs, surprised_imgs) = getData(dataset_used)  # changed

def plotPie():

    num_of_imgs = [len(angry_imgs), len(happy_imgs), len(neutral_imgs), len(sad_imgs), len(surprised_imgs)]
    print(num_of_imgs)
    activities = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

    plt.pie(num_of_imgs, labels=activities, startangle=90, autopct='%.1f%%')
    plt.title('Number of images in each category:\n Angry: {Angry} Happy: {Happy} Neutral: {Neutral} Sad: '
              '{Sad} Surprised:{Surprised}'.format(Angry=num_of_imgs[0], Happy=num_of_imgs[1],
                                                   Neutral=num_of_imgs[2], Sad=num_of_imgs[3],
                                                   Surprised=num_of_imgs[4]))
    plt.show()

plotPie()  # [7612, 10633, 7412, 7736, 7772]

def horizontalFlip(rootpath, emo, dataset):
    i = 0
    for img in os.listdir(dataset):

        if img == '.DS_Store':
            continue

        filename, ext = os.path.splitext(img)

        rootpath = os.path.join(fileOrigin, rootpath)
        img = os.path.join(rootpath, img)

        # Read PIL images
        im = Image.open(img)
        # im.show()
        im = ImageOps.mirror(im)
        im.save('{root}/{filename}_mirror{ext}'.format(root=rootpath, filename=filename, ext=ext))
        # im.show()


'''
# Convert Pil to numpy arrary
im = Image.open(img) 
# img here is the imagepath
arr = np.array(im)
# convert numpy array to pil image
im = Image.fromarray(arr)
'''


def randomNoise(rootpath, emo, dataset):
    i = 0
    for img in os.listdir(dataset):
        if img == '.DS_Store':
            continue
        filename, ext = os.path.splitext(img)

        rootpath = os.path.join(fileOrigin, rootpath)
        img = os.path.join(rootpath, img)

        # Convert Pil to numpy arrary
        im = Image.open(img)  # --> original image
        # im.show()

        arr = np.array(im)
        noisyArr = arr + sk.util.random_noise(arr)
        im = Image.fromarray(noisyArr.astype('uint8'))  # Image after adding noise

        im.save('{root}/{filename}_noise{ext}'.format(root=rootpath, filename=filename, ext=ext))
        # im.show()


def contrastBrightness(rootpath, emo, dataset, alpha, beta):
    i = 0
    for img in os.listdir(dataset):
        if img == '.DS_Store':
            continue
        i += 1
        filename, ext = os.path.splitext(img)

        rootpath = os.path.join(fileOrigin, rootpath)
        img = os.path.join(rootpath, img)

        # Convert Pil to numpy arrary
        im = Image.open(img)  # --> original image
        #im.show()

        arr = np.array(im)
        new_arr = np.zeros(arr.shape, arr.dtype)
        for w, row in enumerate(arr):
            for h, col in enumerate(row):
                for c, color in enumerate(col):
                    new_arr[w, h, c] = np.clip(arr[w, h, c] * alpha + beta, 0, 225)

        im = Image.fromarray(new_arr.astype('uint8'))

        im.save('{root}/{filename}_contrast{ext}'.format(root=rootpath, filename=filename, ext=ext))
        #im.show()


def dataAug(rootFilePath):
    '''Copy the original cropped image directory to the other directory and
    do data augmentation on the original directory to update numbers of images.
    '''
    copy_imgFolder_path = '{root}/ori_{imgDir}'.format(root=fileOrigin, imgDir=rootFilePath)
    # Only copy at the very beginning
    if not os.path.exists(copy_imgFolder_path):
        os.mkdir(copy_imgFolder_path)
        copy_tree('{root}/{imgDir}'.format(root=fileOrigin, imgDir=rootFilePath), copy_imgFolder_path)

    horizontalFlip('{root}/Angry'.format(root = rootFilePath), 'Angry', angry_dataset)
    horizontalFlip('{root}/Sad'.format(root = rootFilePath), 'Sad', sad_dataset)
    horizontalFlip('{root}/Surprised'.format(root = rootFilePath), 'Surprised', surprised_dataset)
    # horizontalFlip('{root}/Happy'.format(root=rootFilePath), 'Happy', happy_dataset)

    randomNoise('{root}/Angry'.format(root = rootFilePath), 'Angry', angry_dataset)
    randomNoise('{root}/Sad'.format(root = rootFilePath), 'Sad', sad_dataset)
    randomNoise('{root}/Surprised'.format(root = rootFilePath), 'Surprised', surprised_dataset)
    # randomNoise('{root}/Happy'.format(root=rootFilePath), 'Happy', happy_dataset)

    alpha, beta = 1.2, 0.5

    contrastBrightness('{root}/Angry'.format(root = rootFilePath), 'Angry', angry_dataset,alpha, beta)
    contrastBrightness('{root}/Sad'.format(root = rootFilePath), 'Sad', sad_dataset, alpha, beta)
    contrastBrightness('{root}/Surprised'.format(root = rootFilePath), 'Surprised', surprised_dataset, alpha, beta)
    # contrastBrightness('{root}/Happy'.format(root=rootFilePath), 'Happy', happy_dataset,  alpha, beta)


'''
This file is mainly used to create suitable inputs based on dataset. 
The input will be format into tensors that can be shaped and batched
'''

# Defines transform to be a normalized to have avg = 0.5 and std = 0.5

def getDataLoader():
    seed = 1
    train_split, val_split, test_split, overfit_split = 0.7, 0.2, 0.1, 30

    (_, _, _, _, _, angry_imgs, happy_imgs, neutral_imgs, sad_imgs, surprised_imgs) = getData(dataset_used)
    num_of_imgs = [len(angry_imgs), len(happy_imgs), len(neutral_imgs), len(sad_imgs), len(surprised_imgs)]
    min_num_img = min(num_of_imgs)
    print(min_num_img)

    # begin_idx is the index of which index will the last one belong to this category
    begin_idx = [0, num_of_imgs[0], num_of_imgs[0] + num_of_imgs[1], num_of_imgs[0] + num_of_imgs[1] + num_of_imgs[2],
                 sum(num_of_imgs) - num_of_imgs[-1]]
    end_idx = [begin_idx[0] + min_num_img, begin_idx[1] + min_num_img, begin_idx[2] + min_num_img,
              begin_idx[3] + min_num_img, begin_idx[4] + min_num_img]

    num_of_imgs = [(end_idx[i] - begin_idx[i]) for i in range(len(end_idx))]
    activities = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprised']

    '''
    # Used for plotting distribution
    plt.pie(num_of_imgs, labels=activities, startangle=90, autopct='%.1f%%')
    plt.title('Number of images in each category:\n Angry: {Angry} Happy: {Happy} Neutral: {Neutral} Sad: '
              '{Sad} Surprised:{Surprised}'.format(Angry = num_of_imgs[0], Happy = num_of_imgs[1],
                                                      Neutral = num_of_imgs[2], Sad = num_of_imgs[3], Surprised = num_of_imgs[4]))
    plt.show()
    '''
    indice, train_indice, val_indice, test_indice, overfit_indice = [], [], [], [], []
    for idx, i in enumerate(begin_idx):
        random.seed(seed)
        sub_idx = []
        for j in range(i, end_idx[idx]):
            sub_idx.append(j)
        random.shuffle(sub_idx)
        train_indice += sub_idx[0:int(train_split * min_num_img)]
        val_indice += sub_idx[int(train_split * min_num_img):int((train_split + val_split) * min_num_img)]
        test_indice += sub_idx[int((train_split + val_split) * min_num_img)::]
        overfit_indice += sub_idx[0:overfit_split]
        indice += sub_idx

    # print(len(train_indice))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    all_input_data = torchvision.datasets.ImageFolder(root=os.path.join(fileOrigin, dataset_used),
                                                      transform=transform)
    # all_data_loader = torch.utils.data.DataLoader(all_input_data, batch_size=4, shuffle=True, num_workers=2)

    balanced_all_input_data = torch.utils.data.Subset(all_input_data, indice)
    train_input_data = torch.utils.data.Subset(all_input_data, train_indice)
    val_input_data = torch.utils.data.Subset(all_input_data, val_indice)
    test_input_data = torch.utils.data.Subset(all_input_data, test_indice)
    overfit_input_data = torch.utils.data.Subset(all_input_data, overfit_indice)
    return balanced_all_input_data, train_input_data, val_input_data, test_input_data, overfit_input_data

# Aid function to plot normalized images haven't changed yet
# For showing out augmented images
def getDataLoader_test():
    seed = 1
    train_split, val_split, test_split, overfit_split = 0.7, 0.2, 0.1, 10
    (_, _, _, _, _, angry_imgs, happy_imgs, neutral_imgs, sad_imgs, surprised_imgs) = getData(dataset_used)
    num_of_imgs = [len(angry_imgs), len(happy_imgs), len(neutral_imgs), len(sad_imgs), len(surprised_imgs)]
    min_num_img = min(num_of_imgs)
    print(min_num_img)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    all_input_data = torchvision.datasets.ImageFolder(root=os.path.join(fileOrigin, dataset_used),
                                                      transform=transform)
    all_data_loader = torch.utils.data.DataLoader(all_input_data, batch_size=1, shuffle=True, num_workers=2)

    k = 0
    for images, labels in all_data_loader:
        # since batch_size = 1, there is only 1 image in `images`
        image = images[0]
        # place the colour channel at the end, instead of at the beginning
        img = np.transpose(image, [1, 2, 0])
        # normalize pixel intensity values to [0, 1]
        img = img / 2 + 0.5

        plt.axis('off')
        plt.imshow(img)
        plt.show()
        k += 1
    return
# getDataLoader_test()

# label is [0][1]
#dataAug(dataset_used)
#balanced_all_input_data, train_input_data, val_input_data, test_input_data, overfit_input_data = getDataLoader()

# Generating suitable data input using Pytorch VGG-16
def get_Input_Dataset(pre_model, input_dataset):
    i = 0
    img_list = []
    for i,(img, label) in enumerate(input_dataset):
        print(i)
        img = img.view(-1, img.shape[0], img.shape[1], img.shape[2])
        vgg_out = pre_model.predict(img)
        vgg_out = vgg_out.detach().numpy()
        img_list.append(label)

        if i == 0:
            vgg_input_arr = vgg_out

        vgg_input_arr = np.concatenate((vgg_input_arr, vgg_out))
        i = 1

    vgg_labels = np.asarray(img_list)
    #print(vgg_input_arr.shape, vgg_labels.shape)
    input_data = input_Dataset.vggDataset(vgg_input_arr, vgg_labels)
    return input_data

# Generating suitable data input using Keras VGG-16
def get_Input_Dataset_keras(pre_model, input_dataset):
    i = 0
    img_list = []
    for i,(img, label) in enumerate(input_dataset):
        #print(i)
        img = img.view(-1, img.shape[1], img.shape[2], img.shape[0])
        img = img.detach().numpy()
        #print(img.shape)
        keras_vgg_out = pre_model.predict(img)
        img_list.append(label)

        if i == 0:
            vgg_input_arr = keras_vgg_out

        vgg_input_arr = np.concatenate((vgg_input_arr, keras_vgg_out))
        i = 1

    vgg_labels = np.asarray(img_list)
    #print(vgg_input_arr.shape, vgg_labels.shape)
    input_data = input_Dataset.vggDataset(vgg_input_arr, vgg_labels)

    return input_data

''' # Testing code
from torchvision import datasets, models, transforms
pre_model = models.vgg16(pretrained=True)
new_overfit_input = get_Input_Dataset(pre_model, overfit_input_data)
new_overfit_data_loader = torch.utils.data.DataLoader(new_overfit_input, batch_size=4, shuffle=True, num_workers=2)
for img, label in new_overfit_data_loader:
    print(img.shape, label)
# train_data_loader = torch.utils.data.DataLoader(train_input_data, batch_size=4, shuffle=True, num_workers=2)
'''
