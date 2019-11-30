#the purpose of this file will be used to process the data from AffectNet to perform cropping and assign a name

import numpy as np
import csv
import matplotlib.image as mpimg
from PIL import Image
#first need to get data from csv file with labels

#file location for images: D:\Machine Learning Datasets\Manually_Annotated_Images
#file location for csv files: D:\Machine Learning Datasets

emotion = "Happy" 
csv_file = "D:\\Machine_Learning_Datasets\\"+emotion+"_Labels.csv"
with open(csv_file, 'r') as f:
    reader = csv.reader(f)
    cnt = 0
    #columns are: ['subDirectory_filePath', 'face_x', 'face_y', 'face_width', 'face_height', 'facial_landmarks', 'expression', 'valence', 'arousal']
    
    """
    NOTE: NEED TO EDIT THE EMOTION VARIABLE AFTER EVERY CHANGE

    """
    for row in reader:
        if(cnt == 0):
            cnt +=1
            continue
        try: 
            label = row[6]
            
            pic_path = './Manually_Annotated_Images/'+row[0]      
            
            pic_path = row[0].split('/')
            img = Image.open("D:\Machine_Learning_Datasets\\Manually_Annotated_Images\\" + pic_path[0] + "\\" + pic_path[1] )

            coords = (int(row[1]), int(row[2]), int(row[1])+int(row[3]), int(row[2])+int(row[4]))

            cropped_img = img.crop(coords)
            cropped_img.show()
            cropped_img=cropped_img.resize((64, 64), Image.ANTIALIAS)

            cropped_img_name = "1-"+str(cnt) + "_0_" + emotion
            print("cropped_img_name", cropped_img_name)
            save_img = cropped_img.save('./cropped_pics/'+emotion+"/"+cropped_img_name+".jpg")
            if(cnt == 7500): #7500 represents the number of images being loaded into the dataset
                break
            cnt +=1
        except:
            pass

print("completed processing")
