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

[image1]: ./examples/center_lane_track1.png 
[image2]: ./examples/center_lane_track1_counter-clockwise.png 
[image3]: ./examples/center_lane_track2.png 
[image4]: ./examples/central_camera.png 
[image5]: ./examples/left_camera.png 
[image6]: ./examples/right_camera.png 
[image7]: ./examples/center_lane_track1_flipped.png
[image8]: ./examples/Angle_Data_Distribution_Total.png
[image9]: ./examples/Angle_Data_Distribution_Filtered.png
[image10]: ./examples/history.png 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
* drive.py - for driving the car in autonomous mode
* model.h5 - containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* run2.mp4 - A video recording of the vehicle driving autonomously at least one lap around the track one.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My models is based on reference from Nvidia's convolutional neural network for self-driving cars from the paper End to End Learning for Self-Driving Cars and Comma.ai's steering angle prediction model. My model consists of a convolution neural network based on Nvidia's model architecture(model.py lines 125-142) and also I added a Cropping2D layer (line 126) to crop the input image and added some dropout layers to generalize the model better.

The model includes RELU layers to introduce nonlinearity (code line 129-133, 136, 138, 140), and the data is normalized and mean centered in the model using a Keras lambda layer (code line 128).

#### 2. Attempts to reduce overfitting in the model

The model contains four dropout layers in order to reduce overfitting (model.py lines 135, 137, 139, 141). 

The model was trained and validated on different data sets from both track1, track2 driving data, and driving data from counter-clockwise on track1, to ensure that the model was not overfitting (code line 154-185). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 179).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving on track1, recovering from the left and right sides of the road on track1, counter-clockwise driving on track1 for one lap, driving data for one lap on track2.

I also used multiple camera's images, flipped images and steering measurements for data augmentation.And also 90% of driving data with 0 degree steering angle were filtered.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create a Convolutiona neural network that predicts the steering angles from the images. Model was created based on refernces from Nvidia's convolutional neural network for self-driving cars and Comma.ai's steering angle prediction model.

My first step was to use a convolution neural network model similar to the Nvidia's model, I thought this model might be appropriate because they have succeeded on real cars.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by using some dropout layers between fully connected layers.

The final step was to run the simulator to see how well the car was driving around both track one and track two. I found there was a left turn bias on track one, so I collected counter-clockwise laps data around the track one and used image flipping technique to overcome this obstacle.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 125-142) consisted of a convolution neural network with the following layers and layer sizes.

model = Sequential()
model.add(Cropping2D(cropping=((50, 20), (0, 0)),input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 127.5 - 1.0))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(50, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded four laps driving in a counter-clockwise direction on track one. Because track one has a left turn bias, if I only drive around the first track in a clockwise direction, the data will be biased towards left turns. Driving counter-clockwise is one way to combat the bias and is also like giving the model a new track to learn from, so the model will generalize better. Here is an example image of driving counter-clockwise:

![alt text][image2]

Then I repeated this process on track two in order to get more data points to generalize the neural network model better. Here is an example image of center lane driving in track two:

[alt text][image3]

To augment the data sat, I also flipped images and angles thinking that this would help with the left turn bias. For example, here is an image that has then been flipped(using numpy.fliplr() function to do this, model.py lines 93-106):

```sh
import numpy as np
image_flipped = np.fliplr(image_original)
steering_angle_flipped = -steering_angle
```

![alt text][image1]
![alt text][image7]

I also used multiple cameras' images. The simulator captures images from three cameras mounted on the car: a center, right and left camera. The following image shows a bird's-eye perspective of the car. From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. And from the right camera's perspective, the steering angle would be larger than the angle from the center camera(model.py lines 70-89).

I chose steering_correction=0.2, and used following code segments to calculate multiple cameras' steering angles.

```sh
left_angle = center_angle + steering_correction
right_angle = center_angle - steering_correction
```
Below are example images of Central, Left and Right camera's

![alt text][image4]
![alt text][image5]
![alt text][image6]

After the collection process, I had around 32000 number of data points from both tracks. The distribution of steering angles is shown below:

![alt text][image8]

From the figure above, we can figure out the the driving data samples with 0 degree steering angle were overrepresented, and the dataset is too imbalanced. So I filtered about 90% driving data samples with 0 degree steering angle. After that, the distribution of steering angles is shown below:

![alt text][image9]

I then preprocessed this data by a lambda layer to parallelize image normalization. The lambda layer will also ensure that the model will normalize input images when making predictions in drive.py. 

I finally randomly shuffled the data set and put 20% of the data into a validation set and used generator function to feed data into the model.(model.py lines 173-186)

I used this training data for training the model. The validation set helped determine if the model was overfitting or underfitting. The ideal number of epochs was 16 as evidenced by the following image which showed the training history. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![alt text][image10]
