# MojiMi

Authors: Eric Li and Kailin Hong

Background Info
- Flickr High Quality Dataset used: 


Set Up:
- note that the preprocessed images and labels are not loaded into the github model yet

To Begin Labelling:
- run the FaceLabeling.py file with "python FaceLabeling.py"
  -> please note that this will look for a file called "labels.csv" to store the labels; you will need to create this file separately on
     your local device
  -> cropped pictures will be stored in the "cropped_pics" folder, which you will also need to create yourself
  -> within the FaceLabeling.py file, there is a loop which considers a range. This range corresponds to the images being processed.
     Please do not set this range to large (~50 per time) to ensure enough error checking.

While the general information is as how the master branch introduced, code for process images from the Flickr Faces HQ Dataset and AffectNet Dataset are combined into file FaceLabelling.py

After obtaining images, they will be preprocessed by running getDataLoader() in the InputManager.py. This function will perform data augmentation and normalize and transform the image into desired tensor format to be compatible with ECNN -- our neuron network. Current version ECNN model can be found in CNNModel.py. 

Training code is in the main.py, the model can be trained and saved to location: savepath by running code as shown below:

Balanced_all_dataset, train_dataset, val_dataset, test_dataset, overfit_dataset = inputManager.getDataLoader()
model = Model.ECNN()
train(model, train_dataset, val_dataset, lr = lr, batch_size = batch_size, num_epoch= num_epoch, save = savepath)
 
  
  




