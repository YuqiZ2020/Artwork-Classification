# Artwork-Classification
Artwork Classification: Using Machine Learning Models to Classify Artwork Types

## Data Sources and Data Preprocessing

The dataset is collected from the Rijksmuseum (Amsterdam, the Netherlands) via its own public online system. It is a large, diverse and open dataset of art objects to support and evaluate automatic art classification and retrieval techniques. It is a set of over 110,000 objects consisting of digital images. For this project, over 4,000 images are selcted over the whole datasets. The works of art date from ancient times, medieval ages and the late 19th century. We randomly choose around 1000 images from each of the 4 classes to make our datasets class-balanced. We cropped the images to 512 * 512 with padding. This is because most of the CNN models take square image inputs.

## Baseline

We included a basic CNN model as a baseline. The model has three layer, each layer consisting of convolutional layer of kernel_size=3, stride=1, padding=1 and a max pooling layer. The channels are increased from 3 to 12 to 24. These choices are common practices of constructing CNN models. The detailed code is shown in Basic_CNN.ipynb with accuracy 62% and a prediction matrix.  

## Models

### Alexnet
AlexNet is a convolutional neural network (CNN) that was trained on the ImageNet dataset and won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC). The AlexNet architecture consists of five convolutional layers and three fully-connected layers. It uses the ReLU activation function between the convolutional layers, ross-entropy (logistic loss) as the objective function, and the Adam optimizer for gradient descent.

We used pytorch's implementation of AlexNet as a template (https://pytorch.org/vision/main/_modules/torchvision/models/alexnet.html), and tuned hyperparamters such as the dimension of image, image preprocessing procedure, batch sizee, learning rate, and the number of epoches.


### VGG and VGG with Transfer Learning
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. (Source: https://en.wikipedia.org/wiki/Transfer_learning)

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task which is related to original task.It is quite popular in deep learning where pre-trained models are used as the starting point on Computer Vision (CV) and Natural Language Processing (NLP) tasks. This is very helpful approach as it saves lots of time, resoureses. This way one can avail benefit of using complex models developed by others as start point and on top of it develop another.

Summary of how to design and implement VGG and TL for this project

    1.1 Prepare dataset: Manipulate original dataset into two folders: test and train. 
        a. Under each folder there are four subfolders corresponding to four catagories. 
        b. 80:20 ration is chosen here, the total test cases are 2971 and the total validation cases are 1275.
    1.2 Store images information.
        From "Preparing training/validation images data" part to transfer image information into designed array.
    1.3 Build default VGG model with transfer leaning.
      Some declarations:
        a. We want to carry weights as it was in original model, so we are carring weights = 'imagenet'.
        b. The very first layer is input layer which accept image size = (224, 224, 3).
        c. We want to change the last layer as we have 4 class classificatoin problem. So, we will not include top layer.
        d. Also, we will not train all the layers except the last one as we will have to train that. 
    1.4 Build custom VGG16 model.
      1.4.1 We setup input layer. 
      1.4.2 We removed top (last) layer.
      1.4.3 We flatten the last layer and added 1 Dense layer and 1 output layer.
      Some declarations:
        a. The first layer is having image size = (224,224,3) now as we defined.
        b. The folloiwng 2 top (last) layers in original VGG16 are not the part of our customized layer as we set include_top=False.
    1.5 Compile and train the model.
    1.6 Case display and discussion.
      Two cases are included here as showing the step for parameter tuning.
      Some discussions:
      1.6.1- Case 1
      a. For lr = 1 and batch_size = 64, after 10 epoch, the accuracy for training set is 99.96% and accuracy for validation set is  over 91%. 
      b. VGG-16 could handle this dataset well since the final classification result is well-calibrated.
      1.6.2- Case 2
      Case 2 is a relative coarse case with a larger learing rate and smaller batch size.
      a. The accuracy for training set is 99.2% and accuracy for validation set is over 90%. 
      b. For the catalory of "photograph", its validation performance is poorer than the rest three catalories relatively: 259/(259+49+17+38) = 71.35%.
      c. More discussions for part b:
          c.1 Background of "photograph" and "photomechanical print" are much similar.
          c.2 Both of them are light yellow and the resolution for "photo" is not as high as "painting" and "drawing". 
          c.3 This relative low performance is due to intrinsic image limitations.
          c.4 It might be hard to dampen the effect and enhance the validition accuracy from modify the model itself. 
          c.5 The latent solution might be pre-processing the dataset as extracting principal component for each artpieces.
### Autoencoder for Interpretable features
All code related with the interpretable key features is in AutoencoderSVD.ipynb. Summary of AutoencoderSVD.ipynb

    1.1 Prepare dataset and store images information.
    1.2 Build autoencoder model.
    1.3 Compile and train the model.
    1.4 Form the latent matrix using encoder from autoencoder
    1.5 Perform PCA/SVD on the latent matrix
    1.6 Visualize the principal components using decoder from autoencoder 
    
Reference: https://www.kaggle.com/code/cdeotte/dog-autoencoder
