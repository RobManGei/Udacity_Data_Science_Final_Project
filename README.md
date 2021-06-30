# Udacity_Data_Science_Final_Project
The repository for the final project of the Udacity Data Science Course. Most of the data is provided by Udacity here https://github.com/udacity/dog-project.git. I added the code in the jupyter notebook.

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

### Problem statement
In order to identify the breed of a dog given an image, a Convolutional Neural Network shall be used.

In this notebook, an algorithm is created that accepts any user-supplied image as input. If a dog is detected in the image, it will provide an estimate of the dog's breed. If a human is detected, it will provide an estimate of the dog breed that is most resembling. In order to do so, several steps are performed:

### Code walkthrough
Firstly, a dog dataset as well as a human dataset is imported. The dog dataset contains training, validation and test images as well as a list of dog breeds (133 breeds). The human dataset contains 13233 images of human faces. 

Secondly, a face detector is created to detect faces in an image. It is an Haar feature-based cascade classifier that is taken from OpenCV. The detector detects a face in all of the 100 sample human images but unfortunately in 11% of the sample dog images a human face is also detected.

![image](https://user-images.githubusercontent.com/65665840/123930804-5b95e400-d990-11eb-9246-073a6c50007c.png)

The project asks if this algorithmic choice necessitates that we communicate to the user that we accept human images only when they provide a clear view of a face (otherwise, we risk having unneccessarily frustrated users!). In your opinion, is this a reasonable expectation to pose on the user? If not, can you think of a way to detect humans in images that does not necessitate an image with a clearly presented face?

I think that is a fairly reasonable expectation for this purpose. However, it would be beneficial if you could use suboptimal pictures. Other possibilities could include other trained models. There are models for profile pictures or upper body detection.

I also tried the default people detector from HOGCV (https://docs.opencv.org/4.5.2/d5/d33/structcv_1_1HOGDescriptor.html) but as it is optimized to detect humans in total, it does not perform well in this task.

A promising approach is described here: https://machinelearningmastery.com/how-to-perform-face-detection-with-classical-and-deep-learning-methods-in-python-with-keras/ It uses deep learning based on the paper “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.” That seems to be performing quite well. I tried to implement it here but installing the package mtcnn caused problems with the installed version of numpy.

In order to detect dogs better, a pre-trained ResNet-50 model to detect dogs in images is used. After some data transformation in order to use tensorflow with 4D tensors from the image paths, a dog detector is created.

![image](https://user-images.githubusercontent.com/65665840/123931650-fee6f900-d990-11eb-85c7-1769aad2dc2c.png)

It can be seen that it performs well (100% detection rate) with dog images as well as with not detecting dogs in the human images (0% detection).

The next step is to create a CNN model from scratch in order to classify the dogs by breed. In order to do so, a sequential keras model is crated:

![image](https://user-images.githubusercontent.com/65665840/123932174-7cab0480-d991-11eb-9563-426c0aad868a.png)

I used three convolutional layers and images are always pooled afterwards to reduce size and account for neighboring pixels. Afterwards it is pooled by global average. Finally, it is densed to 133 (using softmax) as we have so many dog breed categories. The model is compiled and trained. It is optimized using RMS over 10 epochs. The model with the smallest validation lost is stored. This model is tested then an achieves a test accuracy of 3.1% which is better than the random classification and that was the main goal here.

![image](https://user-images.githubusercontent.com/65665840/123933303-897c2800-d992-11eb-90be-0c41af799669.png)

In the next step, a pre-trained model is used (VGG-16 model as a fixed feature extractor). A Global Average Pooling Layer and a densing layer is added. After optimizing this model using RMS over 20 epochs, it achieves an accuracy of 39.5%.

In order to further improve accuracy, another pre-trained model (ResNet-50 bottleneck features) is used.  I added a GAP layer, a dropout layer to randomly drop neurons in order to prevent overfitting and then a densing layer to the 133 categories we have.

![image](https://user-images.githubusercontent.com/65665840/123934179-51291980-d993-11eb-89c8-ef617410c1a9.png)

I used different optimizers and found that the Adam optimizer performs best. Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments. 

![image](https://user-images.githubusercontent.com/65665840/123934534-abc27580-d993-11eb-885e-d16605d3fa82.png)

With an optimization over 20 epochs, the model archieves an accuracy of 81.5%.

![image](https://user-images.githubusercontent.com/65665840/123934670-cd236180-d993-11eb-901d-abbc466aaadc.png)

With this model the doog breed detection algorithem is created. It takes a path to an image and returns a message and prints the image. 

![image](https://user-images.githubusercontent.com/65665840/123934781-eaf0c680-d993-11eb-8b13-4dec3185b073.png)

The output is a bit betteer than expected. Humans are detected properly and the dogs are categoriezed correctly. A plane is detected not to be a human or a dog. Only flaw, in the drone picture, a face was detected. Possible points of improvement are: 1) Training of the model on a larger data set. For this, random transformations of the training data could be used. 2) Fine-tune the model with additional layers. 3) Fine-tuning of the model through fitting and adapting the pre-trained model as well.

![image](https://user-images.githubusercontent.com/65665840/123935152-402cd800-d994-11eb-905f-ad68dfa041bc.png)

![image](https://user-images.githubusercontent.com/65665840/123935203-4b800380-d994-11eb-9b9c-da6208c227cd.png)

![image](https://user-images.githubusercontent.com/65665840/123935246-55096b80-d994-11eb-9265-05ac23396a55.png)

![image](https://user-images.githubusercontent.com/65665840/123935008-212e4600-d994-11eb-9c51-58d70090ac90.png)

![image](https://user-images.githubusercontent.com/65665840/123935056-2b504480-d994-11eb-8d5b-6f413c165521.png)

![image](https://user-images.githubusercontent.com/65665840/123935101-34d9ac80-d994-11eb-90be-9f180aba5c0a.png)
