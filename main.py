#main.py will be the main function of the program in which functions will be run and called from

#config Azure API
import data_processing.FaceAPIConfig as config
import argparse
from PIL import Image
import json
import requests
import urllib 
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
subscription_key, face_api_url = config.config()

headers = {
    'Content-Type':'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false'
}

#get rectangular coordinates from json
def getRectangle(faceDictionary):
    try: 
        rect = faceDictionary['faceRectangle']
        left = rect['left']
        top = rect['top']
        right = left + rect['height']
        bottom = top + rect['width']
        return (left, top, right,bottom)
    except:
        return False

parser = argparse.ArgumentParser()
parser.add_argument('-image', type=str, default="")
parser.add_argument('-dim', type=int, default=64)
parser.add_argument('-NN', type=str, default="ECNN")

args = parser.parse_args()

pic_path = "./images/"+args.image+".jpg"

#get uploaded image 

#pic_url = args.picURL#"https://cdn.psychologytoday.com/sites/default/files/styles/article-inline-half-caption/public/field_blog_entry_images/2018-09/shutterstock_648907024.jpg?itok=0hb44OrI"
#pic_resource = urllib.request.urlretrieve(pic_url, pic_path)
#img = Image.open(pic_path)
#then need to use Axure to get coordinates
#need to figure out what to do with url/file, how to send fiel to axure API

pic_file = open('./images/'+args.image+".jpg", 'rb')
img = Image.open('./images/'+args.image+".jpg")

response = requests.post(face_api_url, params=params, headers=headers, data = pic_file)

#response = requests.post(face_api_url, params=params, headers=headers, json={"url": pic_url})
face_info = response.json() #get face info in dictionary format

print("face_info", face_info)

cropped_names = []
rect_coor_arr = []
for j, val in enumerate(face_info):
    rect_coor = getRectangle(val)
    rect_coor_arr.append(rect_coor)
    if (rect_coor == False):
        print("face not detected")
        continue
    cropped_img = img.crop(rect_coor)
    cropped_img=cropped_img.resize((args.dim, args.dim), Image.ANTIALIAS)
    cropped_img_name = "cropped_"+args.image+"_"+str(j)+".jpg"
    cropped_names.append(cropped_img_name)
    save_img = cropped_img.save("./images/cropped_pics/"+cropped_img_name)

#cropped_names array now contains array of names of cropped files
#the format is of form: cropped_filename.jpg

emotions = []
print("cropped_names", cropped_names)
for name in cropped_names:
    
    #name is a string of cropped img stored
    cropped_img = Image.open("./images/cropped_pics/"+name)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    cropped_img_tensor = transform(cropped_img)

    cropped_img_tensor = cropped_img_tensor.unsqueeze(0)
    ECNN = Model.ECNN_final()
    ECNN.load_state_dict(torch.load('./models/'+args.NN+".pt"))
    ECNN.eval()
    softmax = nn.Softmax(dim=1)
    prediction = softmax(ECNN(cropped_img_tensor))
    emotions.append(prediction.detach().numpy()[0])

#now need to convert the array of emotions int to emotion values and get emoji

original_pic = Image.open(pic_path)
copy_og = original_pic.copy()

for i in range(len(emotions)):
    ind = int(np.where(emotions[i] == np.amax(emotions[i]))[0][0])
    print("emotions",emotions)
    print("ind", ind)
    emoji = ""
    if(ind == 0):
        emoji = "Angry_Emoji.jpg"
    elif(ind == 1):
        emoji = "Happy_Emoji.jpg"
    elif(ind==2):
        emoji = "Neutral_Emoji.jpg"
    elif(ind==3):
        emoji = "Sad_Emoji.jpg"
    elif(ind==4):
        emoji = "Surprised_Emoji.jpg"
    emoji_pic = Image.open("./emojis/"+emoji)

    emoji_pic = emoji_pic.resize((rect_coor_arr[i][2]-rect_coor_arr[i][0], rect_coor_arr[i][3]-rect_coor_arr[i][1]), Image.ANTIALIAS)
  
    copy_og.paste(emoji_pic,  (rect_coor_arr[i][0], rect_coor_arr[i][1]))


copy_og.save('./images/emoji_pasted/emoji_'+args.image+".jpg", quality=95)