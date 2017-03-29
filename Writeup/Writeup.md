# **Traffic Sign Recognition**

## Writeup

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

[image1]: ./Images/SampleVisualtion.png "Visualization"
[image2]: ./Images/SampleCount.png "Sample Count Pre-Augmentation"
[image3]: ./Images/AugmentedImage.png "Augmented Images"
[image4]: ./Images/SampleCount2.png "Sample Count Post-Augmentation"
[image5]: ./Images/StandardisedImage.png "Image Standardisation"
[image6]: ./Images/ConfusionMatrix_train.png "Confusion Matrix Training"
[image7]: ./Images/ConfusionMatrix_train.png "Confusion Matrix Validation"
[image8]: ./Images/TestImages.png "Test Images"
[image9]: ./Images/TestImagesPredictions.png "Test Images - Predictions"
[image10]: ./Images/TestImagesProbabilities.png "Test Images - Probabilities"

---

### Writeup / README

#### 1. Writeup

You're reading it! and here is a link to my [project code](https://github.com/scrottty/CarND-Traffic-Sign-Classifier-Project)

### Data Set Summary & Exploration

The code for this is under the heading **Step 1: Data Summary & Exploration**
#### 1. Data Set Summary
I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of validations set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory Visualization of the Data Set

Here is an exploratory visualization of the data set. It is a matrix of the different classes with the images selected at random. Randomising the selection of the images allows for good exploration of the dataset each time the code is run

![alt text][image1]
![alt text][image2]

### Design and Test a Model Architecture
The code for this section is under **Step 2: Design and Test a Model Architecture**
#### 1. Data Preprocessing
For preprocessing, standardisation for each image was chosen over the other options, normalisation, grayscale, histogram normalisation and combinations of these. For the most part it was selected through trial and error upon the base model to assess its improvement to the model accuracy. It has advantages over normalisation mostly that it loses less information in the process. Rather than the data being contained by +/-1 it is contained within +/-1 standard deviation meaning greater resolution. It also defends against outliers which can cause the majority or the values to be condensed to a small range in normalisation.

It was also chosen to not convert the images to grayscale as firstly it did not improve on the accuracy of the model and secondly it was thought that the colour of the sign (the combination of values across the 3rd axis) might be a key to the models predicting ability.

Here is an example of a traffic sign image before and after standardisation

![alt text][image5]

#### 2. Augmentation of Data Set
The data set came preprocessed into training, validation and testing sets.

After looking at the number of samples per class it was clear that some classes contained a higher number of samples than others. This could cause various issues for the model namely overfitting to the well represented classes. To mitigate this transformed data was introduced to the data set for the classes that we under represented.

The introduced data was augmented randomly in the following ways:
* Image rotation of a random amount between +/-5ºc
* Image translation of a random amount between +/-2 pixels

It was chosen to do this all randomly to stop any unintended patterns influencing the final model. Initially the data augmentation was of a high magnitude but this caused the models accuracy to decrease. It was though this could be because the base architecture was not complex enough to handle such large variations in data and therefore would be required to be deeper or have larger hidden layers.

This image shows the samples per class after the data Augmentation. My final training set was increased from 34799 to 79197.

![alt text][image4]

Here is an example of an original image and an augmented image:

![alt text][image3]

#### 3. Model architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5x6    	| 1x1 stride, valid padding, output: 28x28x6 	|
| Batch Normalisation	|     											|
| Activation            | RELU                                          |
| Max Pooling	      	| 2x2 stride,  output: 14x14x6  				|
| Convolution 5x5x16    | 1x1 stride, valid padding, output: 10x10x16	|
| Batch Normalisation	|     											|
| Activation            | RELU                                          |
| Max Pooling	      	| 2x2 stride,  output: 5x5x16    				|
| Flatten               | Output: 400x1                                 |
| Fully connected		| Output: 120x1  								|
| Dropout               | Keep Probability: 0.5                         |
| Fully connected		| Output: 84x1  								|
| Dropout               | Keep Probability: 0.5                         |
| Fully connected		| Output: 10x1  								|



#### 4. Model Training, Validating and Testing
To train the model, I used an Adam Optimiser. This was chosen over the gradient descent optimser due to its ability to better self-regulate the learning rate and use momentum to better find gloabl minima effectively allowing it to use a larger step size and minimal tuning.

Hyperparameters such as the learning rate and the batch size were left mostly alone as there did not seem to effect the accuracy or running time of the model massively (for the little I adjusted). Number of epochs was increased however to give the model more time to fit to the training data. This can come with the problem of starting to overfit if left to run too long. This was migtigated by only saving the model when it the training and validation accuracies were at a peak. This meant the model was at its optima of fitting the training set best whilst not overfitting.

#### 5. Model Evaluation
My final model results were:
* training set accuracy of 99.8%
* validation set accuracy of 96.8%
* test set accuracy of 94.8%

Confusion matricies were used to help identify errors in the model and see where improvements could be made. From the images it can be seen that the model has most of it confusion with the different speed signs probably due to the numbers looking similar and all having the ame red ring on the outside. Another interesting take away is the problems the model has with the _Double Curve_ and _Wild Animals Crossing_ signs. This seems to be because of the similarity in the symbols, both having the same distinctive bend. Retrospectively, after playing with the feature maps, it would have been good to run the troublesome images through the feature maps and examined them closer to find out what parts the filter was picking up.

![alt text][image6] ![alt text][image7]

**If an iterative approach was chosen:
What was the first architecture that was tried and why was it chosen?**

The LeNet model tuned for the MNIST dataset was used as starting point. Its starting validation accuracy was around 86% after adjustment for the new input.

**Why did you believe it would be relevant to the traffic sign application?**

The solution for which the model was built was very similar to the traffic sign application. Both had images with relatively a small number of features to pick out (pen line, simple image on sign) rather than say trying to distinguish the details of a picture of a dog to pick its breed. With small adjustment it was able to produce a suitable starting result

**How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**

It can predict the test set with an accuracy of 94.8% which is higher than the requirement of a validation accuracy of 93%. This suggests the base model is fairly good for this application

**What were some problems with the initial architecture?**

The LeNet model was found to overfit the data causing the validation accuracy to be much lower than the testing accuracy

**How was the architecture adjusted and why was it adjusted?**

To prevent the overfitting dropouts were introduced to stop the model from relying too much on certain parts of the data. Two dropouts were used, both with a 50% keep probability. This was quite aggressive but through trial and error was found to be the most effective. The placement of the dropouts, after each fully connected layers, was again through trial and error.

To try and help the model tuning, stability and further reduce the model overfitting, batch normalisation was implemented. This adjusted the data prior to each activation to normalise it about its mean and within +/-1 standard deviation. The idea was from this [paper](https://arxiv.org/pdf/1502.03167v3.pdf) which suggests that it make a strong improvement to the model and its training allowing more aggressive learning rates and reducing the need for regularisation though these were not played with in the model (ran out of time!). Each batch of data is normalised during training whilst during testing the data is normalised about the population mean and standard deviation. Whilst interestingly this wasn't found to improve the validation and testing accuracies too much it did make an improvement the models ability to the test images from the web.

Further improvements to prevent overfitting were noted in the previous sections such as Image Standardisation and the augmentation of data

**Which parameters were tuned? How were they adjusted and why?**

Mostly, with the exception of epochs, the hyper-parameters were left alone. This was mostly because I ran out of time. The parameters that were tuned were:
* Number of epochs
* Magnitude of rotation and translation of images
* The keep probability for the dropout

For the most part these were adjusted through trial and error hoping to increase the validation accuracy and therefore reduce the overfitting of the model.

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**

Adding the dropout and the batch normalisation were good choices. They were both effective in reducing the overfitting of the model and increasing it accuracy

### Test a Model on New Images
The code for this section is in **Step 3: Testing a Model on New Images**
#### 1. Chosen Images

Using a mix of Google image search and Google Street View some images were chosen. I attempted to choose one which would be potentially challenging to the model such as on an angle or with an obstruction

Here are the image found on the web:

![alt text][image8]

Initial thoughts on the images were:
1. 30km/h - This was thought to be challenging because of its low resolution and cloudy appearance. However it did have a fairly good contrast with its background which could be beneficial
2. 30km/h - Same a previous but with poor contrast with the sunlight right on it.
3. 80km/h - Another speed sign to test the model on these as they were the most troublesome from the confusion matrix
4. Ahead Only - Should be a fairly straight forward prediction
5. Vehicles Over 3.5 Metric Tons Prohibited - This was included as it was thought to be a test due to its sharp angle.
6. Vehicles Over 3.5 Metric Tons Prohibited - This had the top of another sign in it which could potentially trouble the model
7. Keep Left - This had a blue colour similar to the colour of the sign in the top of the image which was thought to be potentially challenging
8. No Entry - This should be fairly straight forward a prediction
9. No Entry - This was thought to be challenging as it is partially obstructed by a black object. This could throw the model off
10. Road Narrows on the Right - This was chosen as it was a different shape to the other tested images otherwise should be straight forward

#### 2. Model Predictions

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| 30km/h      		    | 30km/h   									    |
| 30km/h     			| 30km/h 										|
| 80km/h				| 80km/h										|
| Ahead Only	      	| Ahead Only					 				|
| Heavy Vehicles		| Heavy Vehicles      							|
| Heavy Vehicles		| Heavy Vehicles      							|
| Keep Left             | Keep Left      							    |
| No Entry              | No Entry      							    |
| No Entry              | No Entry      							    |
| Road Narrows			| Road Narrows      							|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.8%. Its was remarkable and these honestly where the first images found and effort was made to pick tricky ones. One thing they do all have in common is their brightness and relatively good contrast. Perhaps some digging into the incorrect classifications might show that the brightness of the image was the challenge for the model. A to-do perhaps

![alt text][image9]

#### 3. Softmax Probabilities

With the exception of the first _Vehicles Over 3.5 Metric Tons Prohibited_ sign all of the predictions were 100%. This suggests the images were not as challenging as first thought. For the one that didn't have 100% prediction it still had a high (80%) prediction so was not near a false prediction.

![alt text][image10]
