# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # "Image References"

[image0]: ./examples/hist.png "Hist"
[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/center.jpg "Center driving"
[image3]: ./examples/recover.jpg "Recovery driving"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* video.mp4 recording a autonomous driving on track 1 by the trained neural networks

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

Note: The model is only suitable for driving on track 1

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

I tried Python generator to provide data on the fly. From my perspective, it helps avoid the problem of fitting too much data into memory at once. But it also introduces great time lag to training on GPU becuase at each training cycle, a batch of data needs to be read from hard drive and piped to GPU memory by CPU. In general, I wouldn't recommend use Python generator when data size is not large.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the networks proposed by Nvidia for end-to-end training self driving car. It has 5 convolutional layer, first 3 of which are 5x5 kernel, 1 stride and valid padding. The rest are 3x3 kernel, 2 strides and valid padding. They are all followed by ReLU activation layers. After convolutional layers, I stack 3 fully connected layers, with depth of 100, 50 and 10, respectively. They are also ReLU activated. Finally one dense layer is connected to output single unit as final result.

~~My model is exatly LeNet. It is effective enough to solve autonomous driving on track 1. It has one convolutional layer (6 fitlers, 5x5 kernel size, 1 strides and valid padding) followed by another convolutional layer (16 filters, 5x5 kernel size, 1 strides and valid padding). Each of them uses ReLU activation layer and max pooling (2x2 size and strides and valid padding). After convolutional layer, two fully connected layer of depth 120 and 84, respectively, are joined to combine all feature maps. These two layers also use ReLU activation layer. Finally another fully connected layer of 1 unit outputs final result.~~

Normalization by keras Lambda layer is introduced to normalize date to [-0.5, 0.5]. Cropping layer is also used to focus on road instead of sky and trees.

#### 2. Attempts to reduce overfitting in the model

When training, I find the training loss and validation loss as follows: 

```sh
Epoch 1/10
4748/4748 [==============================] - 493s 104ms/step - loss: 0.1324 - val_loss: 0.0143
Epoch 2/10
4748/4748 [==============================] - 497s 105ms/step - loss: 0.0097 - val_loss: 0.0125
Epoch 3/10
4748/4748 [==============================] - 495s 104ms/step - loss: 0.0072 - val_loss: 0.0105
Epoch 4/10
4748/4748 [==============================] - 499s 105ms/step - loss: 0.0052 - val_loss: 0.0103
Epoch 5/10
4748/4748 [==============================] - 497s 105ms/step - loss: 0.0042 - val_loss: 0.0105
Epoch 6/10
4748/4748 [==============================] - 497s 105ms/step - loss: 0.0033 - val_loss: 0.0104
Epoch 7/10
4748/4748 [==============================] - 498s 105ms/step - loss: 0.0025 - val_loss: 0.0111
Epoch 8/10
4748/4748 [==============================] - 501s 105ms/step - loss: 0.0019 - val_loss: 0.0110
Epoch 9/10
4748/4748 [==============================] - 498s 105ms/step - loss: 0.0014 - val_loss: 0.0109
Epoch 10/10
4748/4748 [==============================] - 499s 105ms/step - loss: 9.2083e-04 - val_loss: 0.0119
```

As you can see, training loss is order of magnitude less than validation less, which is a great manifesto of overfitting. 

To reduce overfitting, I used both dropout and L2 regularization. Training loss is almost the same as validation loss.

```sh
4155/4155 [==============================] - 13s - loss: 0.4236 - val_loss: 0.1731
Epoch 2/10
4155/4155 [==============================] - 11s - loss: 0.1404 - val_loss: 0.1176
Epoch 3/10
4155/4155 [==============================] - 11s - loss: 0.1036 - val_loss: 0.0930
Epoch 4/10
4155/4155 [==============================] - 11s - loss: 0.0849 - val_loss: 0.0791
Epoch 5/10
4155/4155 [==============================] - 11s - loss: 0.0730 - val_loss: 0.0706
Epoch 6/10
4155/4155 [==============================] - 11s - loss: 0.0642 - val_loss: 0.0615
Epoch 7/10
4155/4155 [==============================] - 11s - loss: 0.0574 - val_loss: 0.0560
Epoch 8/10
4155/4155 [==============================] - 11s - loss: 0.0528 - val_loss: 0.0523
Epoch 9/10
4155/4155 [==============================] - 11s - loss: 0.0482 - val_loss: 0.0480
Epoch 10/10
4155/4155 [==============================] - 11s - loss: 0.0452 - val_loss: 0.0447
```

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. I used 5e-4 as the intial learning rate for Adam. I use the default batch size set by keras as 32. Validation split ratio is chosen as 0.3 because it is provided by the instructor. L2 regularization coefficient is set as 1e-3 which turns out to be good. I tune the number of epoch from 10 to 15 because sometimes I think the validation loss can continue decreasing after 10 epoches. Also I use early stopping to prevent unnecessary training.

#### 4. Appropriate training data

Training data was captured when I drive the vehicle on both tracks. I tried to keep the vehicle driving along the center line. (Using mouse to control is a great tip!). When training networks, I augument data by flipping the image and take the negative value of steering angle. I also use both left and right camera images and adjut their steering angles by 0.2.

More importantly, I provide training data of recovering from either left or right boundary to the center line. These data are cruical because it could teach the car how to deal with sharp turns. 

From the steering angle histogram of training data, I found most steering angles are 0. The training data are extremly unbalanced, which would mislead networks to output 0 all the time. To combat this, I manually left out 85% of data whose steering angle is 0.

![alt text][image0]

```python
def get_data():
    # Read driving log data from csv file, augment data by lazy flipping and using left and right images
    samples = []
    delta = 0.2
    keep_zero_ratio = 0.15
    angle_corrections = [0.0, delta, -delta]
    with open('./data/driving_log.csv') as f:
        reader = csv.reader(f)
        for line in reader:
            steering_angle = float(line[3])
            # Only keep 15% of 0 steering angle data
            if (abs(steering_angle) < 1e-3) and (np.random.uniform() > keep_zero_ratio):
                continue

            # Use center, left and right images
            for idx in range(3):
                # Original image
                sample = [line[idx], (steering_angle+angle_corrections[idx]), False]
                samples.append(sample)
                # Flipped image
                sample = [line[idx], -(steering_angle+angle_corrections[idx]), True]
                samples.append(sample)
    return samples
```



### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with a simple neural networks of one hidden layer (ReLu activation) with 128 units. No convolution was involved. The whole point of this network was to build the pipeline of data collection, model training and validation. It turned out this simple networks could drive the car on track 1. But it was extremely bad. Steering angle changed erratically. 

Then I tried LeNet to introduce convolutional neural networks because CNN is better for handling images. With the exact architeture of LeNet, the model could somehow smoothly drive car on track 1. However, I found the training loss is order of magnitude less than validation loss. It was overfitting! I used both Dropout(0.5) and L2 regularization (1e-3 coefficient) to attack overfitting. After architeture modification, the car could smoothly drive along the center line. But it still had problem of going through sharp turns. 

I also noticed car constantly turned left even it was supposed to go straight. Clearly, it was due to unbalanced training data. The training data were all about counter-clockwise driving, where left turn is superior to right turn. To balance training data, I flipped the training image and took additive inverse of steering angle (-steering angle). Besides, the ratio of 0 steering angle data seems to be spiky. So I manually prune 85% of 0 steering angle data. At this time, I realized 10 epoches may not enough because validation loss seemed to continue decrease after 10 epoches. So I changed the number of epoches to 15 and add early stopping using Keras module.

Car behavior was slightly better than previously. But it still could not finish sharp turn. Sharp turn was hard because it requires car recovers from side to center very quickly. There are two strategies I used to combat this problem: 1. using both left and right camera images and adjust steering angle accordingly. 2. collecting more data when I manually recover the car from side to center. The first one makes sense becasuse left and right cameras were mounted off from the center. If we pretend center camera is mounted at left(right) camera position, we need larger(less) steering angle.

Car now could finish laps successfully, even for sharp turns. At this time I also tried cropping the image because top 70 pixels are mostly sky and trees, and bottom 25 pixels are mostly car hood, which are irrelevant to driving behavior.

In the end, I tried Python generator to feed data on the fly and fine tune on data collected from track 2. It showed no substantial improvement.

__Updates:__ As one of reviewers recommends, Nvidia's end-to-end self driving car networks seems superior to LeNet since it has more convolutional layers. After changing to the networks, I found it was indeed better than LeNet because car driving behavior were more smooth.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes. The first layer is normalization layer which can map image data to [-0.5, 0.5]. The second layer is a cropping layer so that top 70 and bottom 25 pixels are discarded because they are useless information. The 3rd, 4th and 5th layers are convolutional layers with 5x5 filters, 1 stride and valid padding followed by ReLu layer. The 6th, 7th layers are also convolutional layers with 3x3 filters, 2 strides and validing padding followed by ReLu layer. The last three layers are fully connected layer of depth 100, 50 and 10, all of which followed by ReLu and drop out layer. The output layer is a fully connected layer to single unit output.

Here is a visualization of the architecture.

![alt text][image1]

#### 3. Creation of the Training Set

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from side back to center so that the vehicle would learn to how to deal with sharp turns. These images show what a recovery looks like.

![alt text][image3]

As shown above, the top 70 and bottom 25 pixels are almost useless because sky and trees will not deteremine driving behavior.
