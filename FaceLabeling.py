import requests
from PIL import Image
import os
import FaceAPIConfig as config
import json
from io import BytesIO
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

with open('ffhq-dataset-v2.json') as json_file:
    pic_infos = json.load(json_file)

pic_url = pic_infos["0"]["image"]["file_url"]

subscription_key, face_api_url = config.config()

headers = {
    'Ocp-Apim-Subscription-Key': subscription_key
}

params = {
    'returnFaceId': 'true',
    'returnFaceLandmarks': 'false',
    'returnFaceAttributes' : 'emotion'
}
print("reached")

#used to write the header
#with open('labels.csv', 'w') as csvFile_header:
#    writer = csv.writer(csvFile_header)
#    writer.writerow(['id', 'label'])
#csvFile_header.close()



for i in range(11039,30000): 
    try:
        pic_url = pic_infos[str(i)]["image"]["file_url"]

        response = requests.post(face_api_url, params=params, headers=headers, json={"url": pic_url})

        face_info = response.json() #get face info in dictionary format
        #get picture from URL
        pic_response = requests.get(pic_url)
        img = Image.open(BytesIO(pic_response.content))

        for j, val in enumerate(face_info):
            rect_coor = getRectangle(val)
            if(rect_coor == False):
                continue
            cropped_img = img.crop(rect_coor)

            #cropped_img=cropped_img.resize((64, 64), Image.ANTIALIAS)
            cropped_img=cropped_img.resize((224, 224), Image.ANTIALIAS)

            
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
            """
            label = {
                'id': cropped_img_name,
                'label': max_val
            }  
            """          

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
            #save_img = cropped_img.save("./cropped_pics/"+emotion+"/"+cropped_img_name+".jpg")
            save_img = cropped_img.save("./cropped_pics_224/"+emotion+"/"+cropped_img_name+".jpg")


            #with open('labels.csv', 'a') as csvFile:   
            #    wr = csv.DictWriter(csvFile, fieldnames=('id', 'label'), lineterminator = '\n')
            #    wr.writerow(label)
    except Exception as e:
        print("except ", e)
        continue
    
    print(i)


            
#csvFile.close()

print("finished running")
