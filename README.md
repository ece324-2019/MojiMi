# MojiMi

Authors: Eric Li and Kailin Hong

Background Info:

Given an intput image containg 0, 1, or >1 people, this project has the goal of being able to detect faces represent in the photo and recognize the corresponding emotion on all detected faces. After the emotions are detected, the program is to return the input photo, but with a corresponding emoji matching the recognized emotion on the detected faces. To detect faces, the Microsoft Azure Face API is used. To recognize facial emotions, the Emotion Classification Neural Net (ECNN) is implemented. 

Datasets Used:

To train the models in this project, two datasets are used: Flickr Faces HQ Dataset and AffectNet. They can be found in the links below:
- Flickr High Quality Dataset used: https://github.com/NVlabs/ffhq-dataset
- AffectNet: http://mohammadmahoor.com/affectnet/



While the general information is as how the master branch introduced, code for process images from the Flickr Faces HQ Dataset and AffectNet Dataset are combined into file FaceLabelling.py

After obtaining images, they will be preprocessed by running getDataLoader() in the InputManager.py. This function will perform data augmentation and normalize and transform the image into desired tensor format to be compatible with ECNN -- our neuron network. Current version ECNN model can be found in CNNModel.py. 

Training code is in the main.py, the model can be trained and saved to location: savepath by running code as shown below:

Balanced_all_dataset, train_dataset, val_dataset, test_dataset, overfit_dataset = inputManager.getDataLoader()
model = Model.ECNN()
train(model, train_dataset, val_dataset, lr = lr, batch_size = batch_size, num_epoch= num_epoch, save = savepath)


Files in this Repo: 

FaceAPIConfig.py: contains configurations for Azure Face API, such as subscription_key and face_api_url

Face_FlickrFacesHQ.py: contains the code used to process images from the Flickr Faces HQ Dataset

Face_AffectNet.py: contains code used to process images from the AffectNet Dataset

Models.py: contains the initial ECNN and baseline model archetectures

To Come:
- zipped file contained trained images
- emoji placement code
- interfacing of submodules


