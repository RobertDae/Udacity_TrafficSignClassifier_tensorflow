# **Traffic Sign Recognition** 

## Writeup Robert Dämbkes



---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image1]: ./TrafficSigns_From_Web/0-Speedlimit_20km_per_h.jpeg "Speedlimit_20km_per_h"
[image2]: ./TrafficSigns_From_Web/10-No_passing_for_vehicles_over_3.5_metric_tons.jpg "No_passing_for_vehicles_over_3.5_metric_tons"
[image3]: ./TrafficSigns_From_Web/11-Right_of_way_at_the_next_intersection.jpg "Right_of_way_at_the_next_intersection"
[image4]: ./TrafficSigns_From_Web/12-Priority_Road.jpg "Priority_Road"
[image5]: ./TrafficSigns_From_Web/14-Stop.jpg "Stop"
[image6]: ./TrafficSigns_From_Web/23-Slippery_Road.jpg "Slippery_Road"
[image7]: ./TrafficSigns_From_Web/27-Pedestrian.jpg "Pedestrian"
[images8]: ./TrafficSigns_From_Web/29-Mixed1_Bicycle_Crossing_ID1_ID29.jpg "Mixed1_Bicycle_Crossing_ID1_ID29"
[images9]: ./TrafficSigns_From_Web/31-Wild_animals_crossing.jpg "Wild_animals_crossing"
[images10]: ./TrafficSigns_From_Web/8-Speedlimit_120km_per_h.jpg "Speedlimit_120km_per_h"

[image20]: ./Images_Writeup/Distribution_Classes_Samples.png "Distribution of samples per class"
[image21]: ./Images_Writeup/Exemplary_Picture_of_Each_Class.png "Exemplary picture of each Class"
[image22]: ./Images_Writeup/Pictures_From_the_Web.png "TrafficSigns found on the web"
[image23]: ./Images_Writeup/Probabilities_5.png "Probabilities of the Classes for Classification"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

Here is a link to my [project code](https://github.com/RobertDae/Udacity_TrafficSignClassifier/blob/master/Traffic_Sign_Classifier.ipynb)
Here is a link to my [results](https://github.com/RobertDae/Udacity_TrafficSignClassifier_tensorflow/blob/master/Traffic_Sign_Classifier.html)

### Data Set Summary & Exploration


I used the pandas library to calculate summary statistics of the traffic signs data set:

Number of training examples = 34799
Number of testing examples = 12630
Number of validation examples = 4410
Image data shape = (32, 32, 3)
Number of classes = 43

#### 2. Dataset summary

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Distribution of the sampels per class][image20]

As you can see the samples are not equaly distrubuted. So for a better lerning of the CNN/ LeNet we have to take care that we use equal amount of training data for each class.

The basic statistics such as images shapes, number of traffic sign categories, number of samples in training, validation and test image sets are presented in the Step 1: Dataset Summary & Exploration section, A Basic Summary of the Dataset subsection.

![Exemplary picture of each class][image21]


### Design and Test a Model Architecture
#### Preprocessing

Image preprocessing can be found in Step 2: Design and Test a Model Architecture section, Pre-process the Data Set (normalization, grayscale, and so forth) subsection.

There are some transformations required to be performed on each image to feed it to the neural network.

Normalize RGB image. It is done to make each image "look similar" to each other, to make input consistent.
Convert RGB image to grayscale. It was observed that neural network performs slightly better on the grayscale images. It also may be wrong observations.
I was also tried to use adaptive histogram equalization for improving the local contrast and enhancing the definitions of edges in each region of an image but it decreased the performance of the network, so only normalization and grayscale conversion were used in the final implementation.

The training set was expanded with more data. The intent was to make a count of samples in each category equal, categories containing a smaller number of samples were expanded with more, duplicate, images. The probabilities to get images from each category during training became equal. It dramatically improved neural network performance. The size of training set became 43 * 2010 = 86430 samples.


#### Model Architecture

The model architecture is defined in Step 2: Design and Test a Model Architecture, Model Architecture subsection. The architecture has 5 layers - 2 convolutional and 3 fully connected. It is LeNet-5 architectureThe model architecture is defined in Step 2: Design and Test a Model Architecture, Model Architecture subsection. The architecture has 5 layers - 2 convolutional and 3 fully connected. It is LeNet-5 architecture with only one modification - dropouts were added between the layer #2 and layer #3, the last convolutional layer and the first fully connected layer. It was done to prevent neural network from overfitting and significantly improved its performance as a result.

Below is the description of model architecture.

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Flatten				| outputs 400									|
| Dropout				| Kee_Prob:0.7									|
| Fully connected		| outputs 120									|
| RELU					|												|
| Fully connected		| outputs 84									|
| RELU					|												|
| Fully connected		| outputs 43									|
| RELU					|												|
| Softmax				| 												|
|						|												|
|						|												|

Note: If i introduce more than one dropout my accuracy of the model is sinking and i have to increase the number of epochs


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer which works better than GradientDescentOptimizer.
The following hyperparameters were defined and carefully adjusted:

````
# learning rate; with 0.001, 0.0009 and 0.0007 the performance is worse 
RATE       = 0.0008

# number of training epochs; here the model stops improving; we do not want it to overfit
EPOCHS     = 30

# size of the batch of images per one train operation; surprisingly with larger batch sizes neural network reached lower performance
BATCH_SIZE = 128

# the probability to drop out the specific weight during training (between layer #2 and layer #3)
KEEP_PROB  = 0.7

# standart deviation for tf.truncated_normal for weights initialization
STDDEV     = 0.01
````

#### Solution Approach

Final figures from my model:

Training Accuracy = 99.1%
Validation Accuracy = 93.4%
Validation Accuracy (on other images from the net): 33%

The trained model correctly classifies traffic signs on the training set in 99.1% cases and on the test set in 93.4% cases. The decimal part mostly depends on the data shuffling that is random. The best result I observed was 95% of correct classifications on the validation set; unfortunately, that model was overfitted and performed worse on other images.

The code can be found in section Step 2: Design and Test a Model Architecture, subsection Train, Validate and Test the Model.

In general, I believe LeNet-5 architecture fits good for this task since there are a lot of kinds of traffic signs that contain letters and symbols (LeNet-5 is good for symbols classification). Also, to improve an accuracy of classifications of traffic signs with speed numbers, like 30 km/h, 70 km/h, additional convolutional layers could be added. It is good because this model allowes you to have several layers that can  be max pooling as well as the fully-connected layers or convolution layers. Also it is easy to introduce dropouts to avoid overfitting.

As an improvement i saw that it seems that it makes sense to introduce drop outs on all of the FullyConnected-layers but then i found the model starts to be more difficult to tune and i would have to increase the number of epochs for reaching a similar accuracy percentage.
Therefore i did not do so.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 10 german traffic signs that I found on the web:

![Speedlimit_20km_per_h][image1] ![No_passing_for_vehicles_over_3.5_metric_tons][image2] ![Right_of_way_at_the_next_intersection][image3] 
![Priority_Road][image4] ![Stop][image5] ![Slippery Read][image6]
![Pedestrian][image7] ![Mixed1_Bicycle_Crossing_ID1_ID29][image8] ![Wild_animals_crossing][image9]
![Speedlimit_120km_per_h][image10] 

The eighth image might be difficult to classify because it is a multi sign image from the streets.
So multiple hits may be correct. Also the second sixth is tricky because some parts of the sign are missing and another sign is placed below it.

The new images will look like this after beeing uniformed
![Uniformed_new_images][image22]




Here are the results of the prediction:


| -----------------------------------------------------------------------------------------------|
|                  PREDICTED                  |                   ACTUAL                         |
|:-------------------------------------------:|:------------------------------------------------:|
| 12              Priority road               | 29            Bicycles crossing                  |
| 31          Wild animals crossing           | 31          Wild animals crossing                |
| 12              Priority road               | 12              Priority road                    |
| 35                Ahead only                | 10 No passing for vehicles over 3.5 metric tons  |
| 11  Right-of-way at the next intersection   | 11  Right-of-way at the next intersection        |
| 35                Ahead only                | 0            Speed limit (20km/h)                |
| 18             General caution              | 27               Pedestrians                     |
| 14                   Stop                   | 14                   Stop                        |
| 38                Keep right                | 23              Slippery road                    |
| 12              Priority road               | 8           Speed limit (120km/h)                |
| ---------------------------------------------------------------------------------------------- |

Accuracy = 0.300 (3 out of 10 Images) This is not very impressiv.


The model was able to correctly guess 3 of the 10 traffic signs, which gives an accuracy of 33%. This is much lower that the self test.
This can be because of an overfitting of the LeNet. One of the reasons could be that the perspective of an image tested is not the same as trained before or more signs come into the view (see the mixed image).  


A point to tune the system would be
1. enhance the preprocessing of the images so that the darkness will disapear and a better contrast will be archieved 
(e.g. cv2.equalizeHist or cv2.creeateCLAHE or a combination of both analyizing first the historam bin value if it is bigger than 40) 

2. enhance the LeNet algo by selecting more dropout layers such as in each of the FullyConnected Layers. This ones must be a lower keep_prob as in the CNN layer. But this is much more difficult to tune and also by introducing more dropouts more epoches are needed (about 50.)




#### Performance on New Images
The prediction on the new images from the internet was not very high (33%).  There may be mistakes on other types of images. With other models, I had a problem with "end of all speed and passing limits" traffic sign classification. Also, the results on the test set were not perfect (93.5%), so, certainly, there are images somewhere on the web that this model will not be able to recognize.

The code can be found in the section Step 3: Test a Model on New Images, Load and Output the Images subsection.

Here are the results of the prediction:
![Modell_Prediction_on_Classes][image23]


good fittet model (ID 31):

| ----------------------------------------------------------------------------------------------- |
|                  Class-ID                   |                   Prediction                      |
|:-------------------------------------------:|:-------------------------------------------------:|
| 31         Wild animals crossing            |              87.300%                              |
| 30            Children crossing             |               1.379%                              |
| 23               Slippery road              |               0.092487%                           |
| 19       Dangerous curve to the left        |               1.303E-08%  ~ 0                     |
| 11  Right-of-way at the next intersection   |               1.17763 E-08% ~ 0                   |
| ----------------------------------------------------------------------------------------------- |

badly fittet but working model(ID 12): it recognizes the correct class but the gaps to the other classes are to small (no 1 /0)

badly fittet not working model (ID 10): here generally the model was not good trained some reinforcement lerning could help here.


So you can see in some images the classification works really good 1 class with nearly 100% and the others are close to 0.
In other pictures the model is not so sure about the class identified so we have more classes that are about 40% - 50%





### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

How can i use this visualization? I assume u should put in one image of the Training set and on of the FC-Layers such as fc0 or fc1 in my model. I was not able to call it correctly. It would be a good insight into the algo in order to see if it is performing good or bad. 


