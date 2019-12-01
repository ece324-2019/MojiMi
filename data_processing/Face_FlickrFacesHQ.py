import requests
from PIL import Image
import os
import FaceAPIConfig as config
import json
from io import BytesIO
import matplotlib.pyplot as plt
import csv
from skimage.transform import resize

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

subscription_key, face_api_url = config.config()

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key
} # for url only

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes' : 'emotion'
}

# For the flickr HQ dataseet
def propcessFlickr():
    with open('ffhq-dataset-v2.json') as json_file:
        pic_infos = json.load(json_file)

    pic_url = pic_infos["0"]["image"]["file_url"]

    for i in range(22308,30000):
        print("seen")
        try:
            pic_url = pic_infos[str(i)]["image"]["file_url"]


            for j, val in enumerate(face_info):
                rect_coor = getRectangle(val)
                if(rect_coor == False):
                    continue
                cropped_img = img.crop(rect_coor)

                cropped_img=cropped_img.resize((64, 64), Image.ANTIALIAS)

                emotion_pred = [val['faceAttributes']["emotion"]['anger'],
                    val['faceAttributes']["emotion"]['happiness'],
                    val['faceAttributes']["emotion"]['neutral'],
                    val['faceAttributes']["emotion"]['sadness'],
                    val['faceAttributes']["emotion"]['surprise'],
                ]

                #[anger, happiness, neutral, sadness, surprise]
                max_val = 0
                for e_i, emotion_val in enumerate(emotion_pred):
                    if(emotion_val > emotion_pred[max_val]):
                        max_val = e_i
                if(max_val == 0):
                    emotion="Angry"
                elif(max_val == 1):
                    emotion="Happy"  
                elif(max_val == 2):
                    emotion="Neutral"
                elif(max_val == 3):
                    emotion="Sad"
                elif(max_val == 4):
                    emotion="Surprised"

                cropped_img_name = str(i) + "_" + str(j) + "_" + emotion
                save_img = cropped_img.save("./cropped_pics/"+emotion+"/"+cropped_img_name+".jpg")

                response = requests.post(face_api_url, params=params, headers=headers, json={"url": pic_url})

                face_info = response.json() #get face info in dictionary format
                #get picture from URL
                pic_response = requests.get(pic_url)
                img = Image.open(BytesIO(pic_response.content))
                getCropImgAndLabel(face_info, img, i)

        except Exception as e:
            print("except ", e)
            continue
        print(i)

def getCropImgAndLabel(face_info, img,i):
    for j, val in enumerate(face_info):

        rect_coor = getRectangle(val)
        if (rect_coor == False):
            continue
        cropped_img = img.crop(rect_coor)

        cropped_img_64 = cropped_img.resize((64, 64), Image.ANTIALIAS)
        cropped_img_224 =cropped_img.resize((224, 224), Image.ANTIALIAS)
        plt.figure(i)

        emotion_pred = [val['faceAttributes']["emotion"]['anger'],
                        val['faceAttributes']["emotion"]['happiness'],
                        val['faceAttributes']["emotion"]['neutral'],
                        val['faceAttributes']["emotion"]['sadness'],
                        val['faceAttributes']["emotion"]['surprise'],
                        ]

        # [anger, happiness, neutral, sadness, surprise]
        max_val = 0
        for e_i, emotion_val in enumerate(emotion_pred):
            if (emotion_val > emotion_pred[max_val]):
                max_val = e_i
        if (max_val == 0):
            emotion = "Angry"
        elif (max_val == 1):
            emotion = "Happy"
            continue
        elif (max_val == 2):
            emotion = "Neutral"
        elif (max_val == 3):
            emotion = "Sad"
        elif (max_val == 4):
            emotion = "Surprised"
        cropped_img_64_name = str(i) + "_" + str(j) + "_" + emotion
        cropped_img_224_name = str(i) + "_" + str(j) + "_" + emotion


        # Chnaging folder so knows which one is processed where
        save_img = cropped_img_64.save(os.path.join(os.getcwd(), "cropped_pics_kh_64/") + emotion + "/" + cropped_img_64_name + ".jpg")
        #save_img = cropped_img_224.save(os.path.join(os.getcwd(), "cropped_pics_kh_224/") + emotion + "/" + cropped_img_224_name + ".jpg")
        im = Image.open(os.path.join(os.getcwd(), "cropped_pics_kh_64/") + emotion + "/" + cropped_img_64_name + ".jpg")
        im.show()

def processKDEF_ext():
    headers = {
        'Content-Type':'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
    }

    i = 0
    for (root,dirs,files) in os.walk('KDEF'):
        print(root)
        i +=1
        root = 'KDEF/AF01'
        for file in files:
            file = '1000366164.jpg'
            if file[4:6] == 'HA' or file[4:6] == 'NE':
                continue

            imgpath = os.path.join(root, file)
            img_data = open(imgpath, 'rb')
            img = Image.open(imgpath)

            img.show()
            response = requests.post(face_api_url, params = params, headers=headers, data = img_data)
            face_info = response.json()
            getCropImgAndLabel(face_info, img, 'KDEF_{i}'.format(i=i))
            #i += 1
            break
        if i ==2:
            break



#processKDEF_ext()

def processOriginalPic_ext():
    headers = {
        'Content-Type':'application/octet-stream',
        'Ocp-Apim-Subscription-Key': subscription_key
    }

    i = 0
    for (root,dirs,files) in os.walk('KDEF'):
        print(root)
        for file in files:
            if file[4:6] == 'HA' or file[4:6] == 'NE':
                continue

            imgpath = os.path.join(root, file)
            img_data = open(imgpath, 'rb')
            img = Image.open(imgpath)
            response = requests.post(face_api_url, params = params, headers=headers, data = img_data)
            face_info = response.json()
            getCropImgAndLabel(face_info, img, 'KDEF_{i}'.format(i=i))
            i += 1
print("finished running")
