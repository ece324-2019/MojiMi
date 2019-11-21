#main.py will be the main function of the program in which functions will be run and called from

#config Azure API
import FaceAPIConfig as config
subscription_key, face_api_url = config.config()

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key
}

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false'
    #'returnFaceAttributes' : 'emotion'
}

#get rectangular coordinates from json
def getRectangle(faceDictionary):
    try: 
        rect = faceDictionary['faceRectangle']
        left = rect['left']
        top = rect['top']
        bottom = left + rect['height']
        right = top + rect['width']
        return (left, top, bottom, right)
    except:
        return False

#several different functions

#need args parse to be able to parse and see the file name it should calll for

import argparse
from PIL import Image
import json
import requests

parser = argparse.ArgumentParser()
parser.add_argument('-image', type=str, default="")
args = parser.parse_args()

pic_path = './images/'+args.image

#get uploaded image 
img = Image.open(pic_path)

#then need to use Axure to get coordinates
#need to figure out what to do with url/file, how to send fiel to axure API

response = requests.post(face_api_url, params=params, headers=headers, json={"url": img})
face_info = response.json() #get face info in dictionary format

cropped_names = []
rect_coor_arr = []
for j, val in enumerate(face_info):
    rect_coor = getRectangle(val)
    rect_coor_arr.append(rect_coor)
    if (rect_coor == False):
        print("face not detected")
        
    cropped_img = img.crop(rect_coor)

    cropped_img=cropped_img.resize((64, 64), Image.ANTIALIAS)

    cropped_img_name = "cropped_"+args.image
    cropped_names.append(cropped_img_name)
    save_img = cropped_img.save("./cropped_pics_testing/"+cropped_img_name)

#cropped_names array now contains array of names of cropped files
#the format is of form: cropped_filename.jpg


emotions = []
for name in cropped_names:
    #name is a string of cropped img stored
    cropped_img = img.open(name)
    #call ECNN to get prediction
    #note: will need to change cropped_img to potentially change to string or img
    #emotions.append(ECNN(cropped_img))


#now need to convert the array of emotions int to emotion values and get emoji

#len(emotions) = len(cropped_names) = len(rect_coor_arr)
for i in range(len(emotions)):
    emoji = None
    if(i == 0):
        emoji = "Angry_Emoji.jpg"
    elif(i == 1):
        emoji = "Happy_Emoji.jpg"
    elif(i==2):
        emoji = "Neutral_Emoji.jpg"
    elif(i==3):
        emoji = "Sad_Emoji.jpg"
    elif(i==4):
        emoji = "Surprised_Emoji.jpg"
    
    emoji_pic = Image.open("./emojis/"+emoji)
