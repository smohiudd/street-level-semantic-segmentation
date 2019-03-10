### Segmenting Street-Level Images with Computer Vision using Tensorflow

Read the full blog post here: <https://saadiqm.com/2019/03/06/computer-vision-streets.html>

For this exercise, I’m working with 100 Google Street View images divided into 80 images for training and 20 images for test. Using so few images will not produce a performant model, but this exercise was mainly to familiarize myself with the general CNN training workflow as well as Tensorflow’s data pipeline.

This post is divided into the following sections:

* Image Labelling (Ground Truth)
* Creating Image Label Masks
* Input data/image pipeline & creating TFRecords
* Building the Model
* Training the Model
* Prediction

The final loss/mIOU charts seem somewhat reasonable for a toy example given that we only have 80 training samples and 20 test samples. Of course we are not expecting to see high performance results with such a small dataset despite some data augmentation.

!['loss/mIOU metric graph'](https://s3-us-west-2.amazonaws.com/smohiudd.github.co/unet-segmentation/model_metrics.png)

Finally, lets predict the output masks given some sample images. The output looks acceptable for images with few classes but fails when predicting many classes and complex representations.

!['prediction image'](https://s3-us-west-2.amazonaws.com/smohiudd.github.co/unet-segmentation/prediction_1.png)

!['prediction image'](https://s3-us-west-2.amazonaws.com/smohiudd.github.co/unet-segmentation/prediction_2.png)

!['prediction image'](https://s3-us-west-2.amazonaws.com/smohiudd.github.co/unet-segmentation/prediction_3.png)

!['prediction image'](https://s3-us-west-2.amazonaws.com/smohiudd.github.co/unet-segmentation/prediction_5.png)
