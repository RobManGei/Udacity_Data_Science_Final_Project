# Udacity_Data_Science_Final_Project
The repository for the final project of the Udacity Data Science Course. Most of the data is provided by Udacity here https://github.com/udacity/dog-project.git. I added the code in the jupyter notebook. A writeup can be found here: https://morpheus161.medium.com/classifying-dog-breeds-using-convolutional-neural-networks-1da9b0e2b820

Welcome to my Convolutional Neural Networks (CNN) project in the Data Scientist Nanodegree! In this project, I created a pipeline that can be used to process real-world, user-supplied images. If an image of a dog is supplied, my algorithm will identify an estimate of the dog’s breed. If an image of a human is supplied, the code will identify the resembling dog breed. If neither is in the image, a message will be displayed.

## Files and Folders:
dog_app.ipynb - the jupyter notebook with the solution code

dog_app.html - the jupyter notebook as html for future reference

extract_bottleneck_features.py - python script to ertrackt bottleneck features out of pre-trained CNN models.

bottleneck_features - folder for the bottleneck fetures (too large to store here)

haarcascades - folder for Haar cascade filters

images - folder for sample images

requirements - folder for installation requirements

saved_models - folder for the saved CNN models

## Project Instructions

These are taken from the original repository.

### Instructions

1. Clone the repository and navigate to the downloaded folder.
```	
git clone https://github.com/udacity/dog-project.git or git clone https://github.com/RobManGei/Udacity_Data_Science_Final_Project.git
cd dog-project or cd Udacity_Data_Science_Final_Project
```

2. Download the [dog dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/dogImages`. 

3. Download the [human dataset](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip).  Unzip the folder and place it in the repo, at location `path/to/dog-project/lfw`.  If you are using a Windows machine, you are encouraged to use [7zip](http://www.7-zip.org/) to extract the folder. 

4. Donwload the [VGG-16 bottleneck features](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG16Data.npz) for the dog dataset.  Place it in the repo, at location `path/to/dog-project/bottleneck_features`.

5. (Optional) __If you plan to install TensorFlow with GPU support on your local machine__, follow [the guide](https://www.tensorflow.org/install/) to install the necessary NVIDIA software on your system.  If you are using an EC2 GPU instance, you can skip this step.

6. (Optional) **If you are running the project on your local machine (and not using AWS)**, create (and activate) a new environment.

	- __Linux__ (to install with __GPU support__, change `requirements/dog-linux.yml` to `requirements/dog-linux-gpu.yml`): 
	```
	conda env create -f requirements/dog-linux.yml
	source activate dog-project
	```  
	- __Mac__ (to install with __GPU support__, change `requirements/dog-mac.yml` to `requirements/dog-mac-gpu.yml`): 
	```
	conda env create -f requirements/dog-mac.yml
	source activate dog-project
	```  
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/dog-windows.yml` to `requirements/dog-windows-gpu.yml`):  
	```
	conda env create -f requirements/dog-windows.yml
	activate dog-project
	```

7. (Optional) **If you are running the project on your local machine (and not using AWS)** and Step 6 throws errors, try this __alternative__ step to create your environment.

	- __Linux__ or __Mac__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`): 
	```
	conda create --name dog-project python=3.5
	source activate dog-project
	pip install -r requirements/requirements.txt
	```
	**NOTE:** Some Mac users may need to install a different version of OpenCV
	```
	conda install --channel https://conda.anaconda.org/menpo opencv3
	```
	- __Windows__ (to install with __GPU support__, change `requirements/requirements.txt` to `requirements/requirements-gpu.txt`):  
	```
	conda create --name dog-project python=3.5
	activate dog-project
	pip install -r requirements/requirements.txt
	```
	
8. (Optional) **If you are using AWS**, install Tensorflow.
```
sudo python3 -m pip install -r requirements/requirements-gpu.txt
```
	
9. Switch [Keras backend](https://keras.io/backend/) to TensorFlow.
	- __Linux__ or __Mac__: 
		```
		KERAS_BACKEND=tensorflow python -c "from keras import backend"
		```
	- __Windows__: 
		```
		set KERAS_BACKEND=tensorflow
		python -c "from keras import backend"
		```

10. (Optional) **If you are running the project on your local machine (and not using AWS)**, create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `dog-project` environment. 
```
python -m ipykernel install --user --name dog-project --display-name "dog-project"
```

11. Open the notebook.
```
jupyter notebook dog_app.ipynb
```

12. (Optional) **If you are running the project on your local machine (and not using AWS)**, before running code, change the kernel to match the dog-project environment by using the drop-down menu (**Kernel > Change kernel > dog-project**). Then, follow the instructions in the notebook.

## Description

## Project Definition
### Project Overview

Welcome to my Convolutional Neural Networks (CNN) project in the Data Scientist Nanodegree of Udacity! In this project, I created a pipeline that can be used to process real-world, user-supplied images. If an image of a dog is supplied, my algorithm will identify an estimate of the dog's breed. If an image of a human is supplied, the code will identify the resembling dog breed. If neither is in the image, a message will be displayed. You can find my repro here: https://github.com/RobManGei/Udacity_Data_Science_Final_Project.git

### Problem statement

This project deals with the problem of image classification. Specifically, in order to identify the breed of a dog given an image, a Convolutional Neural Network shall be used. Several approaches were used. A CNN built from scratch was used and assessed and two pre-trained models for image classification (VGG-16 and ResNet-50) were used, adapted and assessed. The architectures of the pre-trained models can be found here https://neurohive.io/en/popular-networks/vgg16/ and here: https://neurohive.io/en/popular-networks/resnet/

With that, an algorithm is created that accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling.
The data sets used in this project can be found here: Dog images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip Human images: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip

## Metrics

In order to assess the performance of the different CNN models created, I used accuracy as a metric. In order to assess the accuracy, the data was split into a training set, a validation set and a test set:

Data files are split into train, test and validation setsAfter a CNN model was created, the model was compiled, fitted to the training set and optimized my minimizing the validation loss:

CNN model is being compiledCNN model is optimized by minimizing the validation lossThe model with the smallest validation loss was then used to assess the accuracy of the prediction using the test set:

Calculation of the accuracy of the CNN model

## Analysis
### Data Exploration and Visualization

In order to get familiar with the data provided, I had a look at the images given. Dog images as well as images of human faces were provided. In total, there were 8351 dog images out of 133 different dog breeds:
Statistics for dog images usedIn addition, there were 13233 images of human faces:
Looking at the data in detail, it can be seen that the train images of the dogs are slightly imbalanced. The figure below shows the distribution of training images over the dog categories.

Distribution of training images over the dog categoriesThe maximum number of images for a dog category is 77 for the Alaskan Malamute and the minimum number of pictures is 26 for the Norwegian Buhund. On average, there are approx. 50 images per dog category with a standard deviation of about 12.
Statistics for the dog training imagesFor the validation set and the test set the situation is similar:
Statistics for dog validation imagesStatistics for dog test images

So, this imbalance could be a factor in the performance of the model. On top of that, the quality of the images differs significantly and in some of the images multiple dogs or even dogs and humans are present. Some images also contain text.
Samples of the training images

## Methodology
### Data Preprocessing
Firstly, the dog dataset as well as a human dataset is imported as described above. The dog dataset contains training, validation and test images as well as a list of dog breeds (133 breeds). The human dataset contains 13233 images of human faces.
For the face detection, the human images were converted from BGR-color images to gray scale images before the detector was used.
Preprocessing of human imagesFor the dog images, the pre-processing steps are the following:
The images are transformed from RGB to PIL format (tf.keras.preprocessing.image.load_img | TensorFlow Core v2.5.0)
The images are converted to tensors
Woth the tensors, the images are converted from RGB to BGR, then each color channel is zero-centered with respect to the ImageNet dataset, without scaling

Preprocessing steps for dog picturesImplementation
In summary, the implementation follows the following methodology:
Implement face detector and dog detector
Implement a CNN model to classify a dog's breed
Train, optimize and validate the model
Assess accuracy of classification of the model
Implement the algorithm as described above

In order to implement the application defined above, a method to distinguish between a dog and a human has to be created. Therefore, a face detector and a dog detector are implemented.
Face detector
The face detector is created to detect human faces in an image. It is an Haar feature-based cascade classifier that is taken from OpenCV (https://docs.opencv.org/4.5.2/db/d28/tutorial_cascade_classifier.html). The detector performs well in human images as it detects a face in all of the 100 sample human images. Unfortunately, in 11% of the sample dog images a human face is also detected. This means, another detector for the dog images is needed.
The project asks if this algorithmic choice is acceptable as it necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unnecessarily frustrated users).
I think that is a fairly reasonable expectation for this purpose. However, it would be beneficial if you could use suboptimal pictures as well. Other possibilities to detect faces could include other classifiers. There are classifiers for profile pictures or upper body detection available as well (https://github.com/opencv/opencv/tree/master/data/haarcascades).
I also tried the default people detector from HOGCV (https://docs.opencv.org/4.5.2/d5/d33/structcv_1_1HOGDescriptor.html) but as it is optimized to detect humans in total, it does not perform well in this task. For example, it detects a human in this picture:
Human detected using HOGCV classifierA promising approach to detect faces using CNN is described here: https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/. It uses deep learning based on the paper "Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks". This approach seems to be performing quite well. I tried to implement it here, but installing the package "mtcnn" caused problems with the installed version of numpy and led to 
incompatibility with other packages used in the project.
Dog detector
In order to detect dogs better, a pre-trained ResNet-50 model for image classification is used (https://keras.io/api/applications/#usage-examples-for-image-classification-models). After some data transformation, in order to use tensorflow with 4D tensors from the image paths, a dog detector is created.
This implementation uses the CNN model to classify a given image and if the prediction is one of the dog breeds, the detector returns true:
Dog detector using pre-trained ResNet50 modelIt can be seen that it performs well (100% detection rate) with dog images as well as with not detecting dogs in the human images (0% false detection) with the given sample.
Results of dog detectorCNN model from scratch
The next step is to create a CNN model from scratch in order to classify the dogs by breed. In order to do so, a sequential keras model is crated:
I used three convolutional layers and images are always pooled afterwards to reduce size and account for neighboring pixels. Afterwards it is pooled by global average. Finally, it is densed to 133 (using softmax) as we have that many dog breed categories. The model is compiled and trained. It is optimized using RMS over 10 epochs. The model with the smallest validation loss is stored. This model is then tested and achieves a test accuracy of 2.9% which is better than the random classification (1/133) and that was the main goal here.
Test accuracy for CNN model from scratchRefinement
In the next step, a pre-trained model is used (VGG-16 model as a fixed feature extractor using bottleneck features (the last activation maps before the fully-connected layers)).
A Global Average Pooling Layer and a densing layer is added. After optimizing this model using RMS over 20 epochs, it achieves an accuracy of 42%.
In order to further improve accuracy, another pre-trained model (ResNet-50 bottleneck features) is used. I added a GAP layer, a dropout layer to randomly drop neurons in order to prevent overfitting and then a densing layer to the 133 categories we have.
I used different optimizers and found that the Adam optimizer performs best (slightly better than RMSprop). Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
With an optimization over 20 epochs, the model achieves an accuracy of 81.9% (compared to 79% with a RMSprop optimizer).
Test accuracy of final modelWith this model the dog breed detection algorithm is created. It takes a path to an image and returns a message and prints the image.
Dog breed classification algorithm

## Results
### Model Evaluation and Validation

As described above, the final CNN model that used transfer learning based on a pre-trained ResNet50 model achieved an accuracy on the test set of 81.5%. This model was validated on the validation set and the one with the smallest validation loss. In order demonstrate the robustness of my model, I performed a k-fold validation (https://neptune.ai/blog/cross-validation-in-machine-learning-how-to-do-it-right). For this I only used the training set and the targets. The dataset was split in 10 folds and the model from above was fitted and optimized with the same parameters. 
The model with the smallest validation loss in each iteration was tested with the original test set. The figure below shows the obtained accuracy values.
k-fold cross validation resultsIt can bee seen, that the test accuracy remains pretty stable and therefore, the characteristics of the model are assumed to be acceptable for this task.

### Justification

As described above, the ResNet50 model was used as it achieved higher accuracy than the VGG-16 model. In addition, different optimizers (RMSprop, Adam, see https://keras.io/api/optimizers/) were used to train the model. The used setup was the one yielding the maximum accuracy on the test set.
The figures below show some results using the final algorithm. The output is a bit better than expected. Humans are detected properly and the dogs are categorized correctly in the samples. A plane is detected not to be a human or a dog. Only flaw, in the drone picture, a human face was detected.

## Conclusion
### Reflection

As I am very interested in deep learning, I took this opportunity in the Data Scientist Nanodegree by Udacity to dig deeper into the topic.
It can be seen, that using transfer learning and pre-trained models that have been trained on a large data set is far superior to building CNN models from scratch (at least in this case). In addition, a lot of fine-tuning and tweaking of the models is possible, for which a lot of experience is helpful. Sill, I am impressed how many different libraries and models exist and that they can be integrated rather easily.
It has to be stated however, that this task and the model used is far from perfect and other images or datasets can give very diverging results.

### Improvement

There are several possible points of improvement for the model. Some of them are: 1) Training of the model on a larger data set. For this, random transformations of the training data could be used. 2) Fine-tune the model with additional layers. 3) Fine-tuning of the model through fitting and adapting the pre-trained model as well.
