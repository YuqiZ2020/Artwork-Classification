# Artwork-Classification
Artwork Classification: Using Machine Learning Models to Classify Artwork Types

## Data Sources and Data Preprocessing

## Baseline

## Models

### Alexnet

### VGG and VGG with Transfer Learning
Transfer learning (TL) is a research problem in machine learning (ML) that focuses on storing knowledge gained while solving one problem and applying it to a different but related problem. For example, knowledge gained while learning to recognize cars could apply when trying to recognize trucks. (Source: https://en.wikipedia.org/wiki/Transfer_learning)

Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task which is related to original task.It is quite popular in deep learning where pre-trained models are used as the starting point on Computer Vision (CV) and Natural Language Processing (NLP) tasks. This is very helpful approach as it saves lots of time, resoureses. This way one can avail benefit of using complex models developed by others as start point and on top of it develop another.

Summary of how to design and implement VGG and TL for this project

    1.1 Prepare dataset: Manipulate original dataset into two folders: test and train. Under each folder there are four subfolders corresponding to four catagories. To be more specific, 80:20 ration is chosen here, the total test cases are 2971 and the total validation cases are 1275.
    1.2 Store images information via "Preparing Training image data/Preparing validation images data" to transfer image information into designed array.
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
