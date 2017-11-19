# **Behavioral Cloning** 



---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture.png "nVidia CNN"
[image2]: ./centerCam.jpg "Center View"
[image3]: ./leftCam.jpg "Left View"
[image4]: ./rightCam.jpg "Right View"
[image5]: ./flipedCam.jpg "Right View"




---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. CNN model architecture 

I have used a model published by nVida self-driving team. I trained the weights of the network to minimize the mean-squared error between the steering command output by the network, and the user steering input, from a recorded session using Udacity simulator. the figure below shows the network architecture, which consists of 9 layers, including a normalization layer, 5 convolutional layers, and 3 fully connected layers. In addition, I have added two dropout layers to prevent overfitting. The first one after the 200 Dense layer, and after the 50 Dense Layer.

![alt text][image1]



#### 3. Model parameter tuning

The model used an Adam optimizer, so the learning rate was not tuned manually. I used 3 epochs to train the model.

Epoch 1/3

9900/9900 [==============================] - 109s - loss: 0.0240 - val_loss: 0.0199

Epoch 2/3

9900/9900 [==============================] - 109s - loss: 0.0184 - val_loss: 0.0234

Epoch 3/3

9900/9900 [==============================] - 110s - loss: 0.0158 - val_loss: 0.0202


#### 4. Appropriate training data

Training and validation data from a recorded three laps session, using the Udacity simulator. The total number of the pictures is around 9900 pictures. The set consists of images collected from three virtual cameras in the simulator, center, left, and right. In addition, I augmented the center pictures by flipping them. 

##### Center Cam
![alt text][image2]


##### Left Cam
![alt text][image3]


##### Right Cam
![alt text][image4]


##### fliped Cam
![alt text][image5]

#### 5. Evaluation the model

Using the same Udacity simulator, the model successfully completed one lap driving without the car getting off the track.  Please check the recorded video file, "video.mp4." 
