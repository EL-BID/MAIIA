# Semantic Segmentation

Semantic Segmentation is the problem of assigning labels to every pixel in an
image. These labels correspond to a fixed number of classifications. There is
no notion of distinct objects.

Typical training data are pairs of matching images and masks. A mask is an array
of classification labels with exactly the dimensions of its corresponding image.

## Overview of network architectures

### Fully Convolutional Networks (FCN)

These networks consist of an encoder-decoder architecture, first producing
feature maps by stacking convolutional and pooling layers which successively
downsample the image size.

The downsampled feature maps must then be upsampled again to produce an array of
output labels the same size as the original image. To accomplish this, transpose
(or sometimes "de-") convolutional layers.

They are named as such since they do not use fully connected layers at all, and
are thus entirely composed of convolution-like operations (so completely
input-size independent).

The original reference can be found here
[Fully Convolutional Networks for Semantic Segmentation][2].

### SegNet

### DeconvNet

### DeepLab-V3+

### U-Net

U-Net comes into play for complex image segmentation problems.
Similar to the architecture of DeepLab, it consists of an encoder and decoder.
More particularly, the whole idea of U-Net is a further development of
Convolutional Neural Networks (CNNs),  which learns the feature mapping of
images in order to make more nuanced feature mapping. However, the CNN approach
doesn’t work well in image segmentation, where we also need to reconstruct an
image from a vector. The decoder of U-Net is responsible for this reconstruction
task. During the encoder phase, we have already learned the feature mapping.
We use the same feature maps that are used for contraction to expand a vector
to a segmented image. Without going further in-depth, the structure of the
U-Net model looks like an "U“, to which it owes its name.

### Deep U-Net

### DeepResNet

### Deep Res U-Net

## Code Samples

### [Deep Residual U-Net (ResUNet)][4]

**Promising!**

### [DeepNetsForEO][1]

**promising place to start?**

This repository contains code, network definitions and pre-trained models for
working on remote sensing images using deep learning.

We build on the SegNet architecture (Badrinarayanan et al., 2015) to provide a
semantic labeling network able to perform dense prediction on remote sensing
data. The implementation uses the PyTorch framework.

### [Robosat][3]

Open-source project with EO semantic segmentation pipelines including U-Net.

No pretrained model yet.

--------------------------------------------------------------------------------

# Object Detection

Object detection is a supervised learning problem which aims to predict the
*positions* (specified by bounding boxes) of objects within an image, and the
*probabilities* of each object of belonging to some of a fixed number of
classifications. The number of objects present does not have to be known a priori.

*Training data consists of images with object categories and bounding boxes
for each instance of that category.*

## Overview of network architectures

Network architectures fall broadly into two categories:

### Two-Shot networks

Here, Two subnetworks exist, one for generating *region
proposals (RP)*, the other for detecting the object/doing *region processing*
on each proposed region. These necessarily have multi-task losses.

#### Region(R-)CNN:

1. Input image is passed to region proposal algorithm.
2. Region proposals done with standard CPU signal processing algorithm
  based on edge-detection. Regions warped to fixed-size for downstream nw.
3. Convolutional layers learn feature maps from each region.
4. SVN branch classifies objects based on feature maps
5. Regression branch simultaneously predicts offset corrections to
  bounding boxes from feature maps

- Uninteresting regions are classified as in background class.
- Softmax classifier (log loss), SVMs (hinge loss), bbox regr. (L2 loss)
- **SLOW**

#### Fast R-CNN:

- Input image is passed to convolutional layers
- Convolutional layers first produce feature maps
- Region proposals again from classical algorithm, but are generated on
  convolutional feature map rather than base image
- Crops from feature maps are warped using "RoI pooling" layer
- Fully-connected layers predict classification scores and bounding-
  box corrections
- Softmax classifier (log loss), bbox regr. (smooth L1 loss)
- **Training ~10x, inference ~20x faster than R-CNN**
- **Bottlenecked by computing region proposals**

#### Faster R-CNN:

- Input image passed to convolutional layers
- Convolutional layers derive feature maps
- Region-Proposal Network predicts region proposals inside network from
  feature maps
- Crops from feature maps are warped using "RoI pooling" layer
- Remaining layers as in Fast R-CNN
- Four way multi-task loss.
  - RPN needs to decide for each proposal is there an object or not an object
    and regress bounding box co-ordinates for each proposal. *Since
    there is no ground truth for each proposed region to train on,
    any time you propose a region with a particular degree of overlap
    (specified by some hyperparameters) with any of the ground truth
    objects, this is considered a positive region proposal and v.v..*
    Binary classifier (log loss) and bounding box regression (LX loss).
  - Final network has to do this again, making final classification
    scores (log loss) and correcting bounding box co-ordinates again (LX
    loss).
- **~10x faster still. Eliminates overhead of doing RP outside network.**

### One-Shot networks:

Fully feed-forward, single pass architectures. No
separate region proposal branch, and so no independent processing for each
potential region. Instead treat as regression problem, making all predictions
at once.

#### YOLO (You Only Look Once):

- Input image divided into coarse grid (e.g. 7x7)
- Within each grid cell, for a set of B base bounding boxes (with different
  dims) and with C classifications, predict:
  - an offset and confidence for object (dx, dy, dh, dw, confidence)
  - classification scores (p_0, p_1, ...)
  Returning a tensor of dimensions 7 x 7 * (5*B + C)
- Goes from input image to tensor of scores with one big conv network.
- Matching ground truth objects with base boxes hairy again and requires some
  hyperparameter-tuned overlap checking.
- **faster than Faster R-CNN**

#### SSD (Single-Shot Detection):
As above with minor tweaks
**faster**

#### DSSD (Deconvolutional SSD)
better than SSD

#### RetinaNet
seems better than all the other single-stages?

## Code samples

**TODO**

-------------------------------------------------------------------------------

# Instance Segmentation

Instance Segmentation is a supervised learning problem which goes a step further
by aiming to predict - instead of bounding boxes - *pixel-wise masks* for each
object, along with the associated *classification probabilities*. Hybrid of
semantic segmentation and object detection. The number of objects present does
not have to be known a priori.

*Training data consists of images with object categories and pixel-wise masks
for each instance of that category.*

## Overview of network architectures

Network architectures fall broadly into two categories:

- **Two-Shot networks:** Two subnetworks exist, one for generating *region
  proposals (RP)*, the other for detecting the object/doing *region processing*
  on each proposed region. These networks have multi-task losses. Architectures
  are usually further factorised into the *backbone* (typically convolutional
  architectures to perform the feature extraction) and the *head* (the
  subnetworks for bounding-box and mask extraction, plus classification).
  - **[Mask R-CNN][10]:**
    - Input image passed to convolutional layers
    - Convolutional layers derive feature maps
    - Region-Proposal Network predicts region proposals inside network from
      feature maps (exactly as in Faster R-CNN)
    - Crops from feature maps are standardised using "RoI *align*" layer, to
      warp proposed regions into the right shape. This method removes the
      quantisation of RoIPool using interpolation. RoIPool quantisation was
      found to introduce misalginments between RoI and extracted features.
    - The RoI'd feature maps are interpeted by fully-connected layers to
      predict a classification (log loss) and box shift (LX loss), while...
    - ...in parallel *an additional semantic segmentation branch uses transpose
      convolutions to predict a pixel-wise one-hot mask of size x * y *
      n_classifications*.
    - Five way multi-task loss.
        - RPN has binary classifier (log loss) and bounding box regression
          (LX loss), as in Faster R-CNN.
        - Final network has to do this again, making final classification
          scores (log loss), correcting bounding box co-ordinates again (LX
          loss), and evaluating the mask fit (average binary log loss on mask
          matching ground truth class)
    - (*Optional:*) can extend classification/box regression branch with e.g.
            human joint co-ordinates to do things like pose estimation
    - **performs incredibly well**


-------------------------------------------------------------------------------

# Change Detection

**under construction - plenty of info in individual paper descriptions in
[References](./REFS.md)!**

## Code samples


[1]: https://github.com/nshaud/DeepNetsForEO
[2]:https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
[3]: https://github.com/mapbox/robosat
[4]:https://github.com/nikhilroxtomar/Deep-Residual-Unet/blob/master/Deep%20Residual%20UNet.ipynb
