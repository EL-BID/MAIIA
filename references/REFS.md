References - Computer Vision for Earth Observation
======================

# General

Below is a list of general resources for deep learning (possibly in remote
sensing), which do not fall into a specific category or pertain to a certain
problem domain.

## Papers

### [A survey on Image Data Augmentation for Deep Learning][157]

Deep convolutional neural networks have performed remarkably well on many Computer Vision tasks. However, these networks are heavily reliant on big data to avoid overfitting. Overfitting refers to the phenomenon when a network learns a function with very high variance such as to perfectly model the training data. Unfor-tunately, many application domains do not have access to big data, such as medical image analysis. This survey focuses on Data Augmentation, a data-space solution to the problem of limited data. Data Augmentation encompasses a suite of techniques that enhance the size and quality of training datasets such that better Deep Learning models can be built using them. The image augmentation algorithms discussed in this survey include geometric transformations, color space augmentations, kernel filters, mixing images, random erasing, feature space augmentation, adversarial training, generative adversarial networks, neural style transfer, and meta-learning. The applica-tion of augmentation methods based on GANs are heavily covered in this survey. In addition to augmentation techniques, this paper will briefly discuss other character-istics of Data Augmentation such as test-time augmentation, resolution impact, final dataset size, and curriculum learning. This survey will present existing methods for Data Augmentation, promising developments, and meta-level decisions for implementing Data Augmentation. Readers will understand how Data Augmentation can improve the performance of their models and expand limited datasets to take advantage of the capabilities of big data.

[157]: https://www.researchgate.net/publication/334279066_A_survey_on_Image_Data_Augmentation_for_Deep_Learning/link/5d20d9f3458515c11c18cb90/download


### [Structured Convolutions for Efficient Neural Network Design][141]

In this work, we tackle model efficiency by exploiting redundancy in the implicit structure of the building blocks of convolutional neural networks. We start our analysis by introducing a general definition of Composite Kernel structures that enable the execution of convolution operations in the form of efficient, scaled, sum-pooling components. As its special case, we propose Structured Convolutions and show that these allow decomposition of the convolution operation into a sum-pooling operation followed by a convolution with significantly lower complexity and fewer weights. We show how this decomposition can be applied to 2D and 3D kernels as well as the fully-connected layers. Furthermore, we present a Structural Regularization loss that promotes neural network layers to leverage on this desired structure in a way that, after training, they can be decomposed with negligible performance loss. By applying our method to a wide range of CNN architectures, we demonstrate "structured" versions of the ResNets that are up to 2× smaller and a new Structured-MobileNetV2 that is more efficient while staying within an accuracy loss of 1% on ImageNet and CIFAR-10 datasets. We also show similar structured versions of EfficientNet on ImageNet and HRNet architecture for semantic segmentation on the Cityscapes dataset. Our method performs equally well or superior in terms of the complexity reduction in comparison to the existing tensor decomposition and channel pruning methods.

[141]: https://arxiv.org/abs/2008.02454

### [Remote Sensing Image Scene Classification MeetsDeep Learning: Challenges, Methods, Benchmarks,and Opportunities][93]

Remote  sensing  image  scene  classification,  whichaims  at  labeling  remote  sensing  images  with  a  set  of  semanticcategories  based  on  their  contents,  has  broad  applications  ina  range  of  fields.  Propelled  by  the  powerful  feature  learningcapabilities of deep neural networks, remote sensing image sceneclassification  driven  by  deep  learning  has  drawn  remarkableattention and achieved significant breakthroughs. However, to thebest of our knowledge, a comprehensive review of recent achieve-ments regarding deep learning for scene classification of remotesensing images is still lacking. Considering the rapid evolution ofthis field, this paper provides a systematic survey of deep learningmethods for remote sensing image scene classification by coveringmore  than  160  papers.  To  be  specific,  we  discuss  the  mainchallenges of remote sensing image scene classification and survey(1)  Autoencoder-based  remote  sensing  image  scene  classificationmethods, (2) Convolutional Neural Network-based remote sensingimage  scene  classification  methods,  and  (3)  Generative  Adver-sarial  Network-based  remote  sensing  image  scene  classificationmethods.  In  addition,  we  introduce  the  benchmarks  used  forremote  sensing  image  scene  classification  and  summarize  theperformance of more than two dozen of representative algorithmson three commonly-used benchmark data sets. Finally, we discussthe  promising  opportunities  for  further  research.

### [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks][86]

Describes a scaling strategy to maximally improve convnet performance at fixed computational resources.
Involves scaling filters, depth and input resolution simultaneously by empirical coefficients.

### [Deep Learning in Remote Sensing: A Review][17]

This covers a broad range of topics beginning with elementary machine learning,
but moving on to EO applications in:

- Hyperspectral imaging
- Synthetic aperture radar (SAR)
- High-resolution satellite images
  - Scene classification
  - Object detection
  - Change detection
- Multimodal data fusion
- 3D reconstruction

### [Deep learning in remote sensing applications: A meta-analysis and review][61]

*TODO*

### [Learning to Reweight Examples for Robust Deep Learning][78]

Seems like a promising method to reweight biased datasets that seriously
outperforms standard techniques like resampling and inverse frequency weighting.


### [Survey of Deep-Learning Approaches for Remote Sensing Observation Enhancement][81]

Comprehensive review of deep-learning methods for the enhancement of remote
sensing observations, focusing on critical tasks including single and
multi-band super-resolution, denoising, restoration, pan-sharpening, and
fusion, among others.

### [Learning to Zoom: a Saliency-Based Sampling Layer for Neural Networks][82]

Proposes a trainable saliency-based sampling layer that selectively upsamples
regions of low-resolution input images trained with access to higher resolution
originals.

### [PSI-CNN: A Pyramid-Based Scale-Invariant CNN Architecture for Face Recognition Robust to Various Image Resolutions][79]

Makes attempt to circumvent dependency on input image resolution on CNN's
performance by proposing PSI-CNN, a generic pyramid-based scale-invariant CNN
architecture which additionally extracts untrained feature maps across multiple
image resolutions, allowing the network to learn scale-independent
information and improving the recognition performance on low resolution images.

Experimental results on the LFW dataset and a CCTV database show PSI-CNN
consistently outperforming the widely-adopted VGG face model in terms of face
matching accuracy.

*looks like this applies during training, downsampling and fusing feature
information from different downsample factors to give tolerance to scale
variation*

### [Deep Residual Learning for Image Recognition][85]

Introduces residual blocks, which learn residual functions with respect to the
layer inputs rather than unreferenced functions. Leads to easier optimisation,
accuracy gains and possibility of much deeper networks.

## General Datasets

### [Zeebruges, or the Data Fusion Contest 2015 dataset][75]

Unlabelled, high quality fusion data including orthophotos and LiDAR point cloud.

LiDAR data at about 10-cm point spacing
Color orthophoto at 5-cm spatial resolution
7 separate tiles - each with a 10000 × 10000 px portion of color orthophoto
(GeoTIFF, RGB), a max 5000 × 5000 px portion of the Digital Surface Model
(GeoTIFF, floating point), and a text file listing the 3D points in XYZI format
[containing X (latitude), Y (longitude), Z (elevation), and I (LiDAR intensity)
information].

See also: [Extremely High Resolution LiDAR and Optical Data: A Data Fusion Challenge][49]

### [GRSS Data and Algorithm Standard Evaluation (DASE) Website][50]

A compendium of data sources and associated algorithm performance league tables
on certain challenges. Covers many of the specific datasets summarised below.

### [Net2Net: Accelerating Learning Via Knowledge Transfer][12]

Promising general transfer learning reference.

## Slides

### [An introduction to  Remote Sensing Analysis  with applications on land cover mapping][80]

A (very recent) blitz through various DL applications including time series
image analysis, Recurrent-CNNs etc.

## Articles

### [Deep learning for remote sensing - an introduction][11]

## Videos:

### [Detection and Segmentation][3]

-------------------------------------------------------------------------------

# Semantic Segmentation

Semantic Segmentation is the problem of assigning labels to every pixel in an
image. These labels correspond to a fixed number of classifications. There is
no notion of distinct objects.

Typical training data are pairs of matching images and masks. A mask is an array
of classification labels with exactly the dimensions of its corresponding image.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md).

## Datasets

### [The Synthinel-1 dataset: a collection of high resolution synthetic overhead imagery for building segmentation][131]

Recently deep learning – namely convolutional neural networks (CNNs) have
yielded impressive performance for the task of building segmentation on large
overhead (e.g. satellite) imagery benchmarks. However, these benchmark datasets
only capture a small fraction of the variability present in real-world overhead
imagery, limiting the ability to properly train, or evaluate, models for
real-world application. Unfortunately, developing a dataset that captures even
a small fraction of real-world variability is typically infeasible due to the
cost of imagery, and manual pixel-wise labeling of the imagery. In this work we
develop an approach to rapidly and cheaply generate large and diverse virtual
environments from which we can capture synthetic overhead imagery for training
segmentation CNNs. Using this approach, generate and publicly-release a
collection of synthetic overhead imagery – termed  Synthinel-1 with full
pixel-wise building labels. We use several benchmark dataset to demonstrate that
Synthinel-1 is consistently beneficial when used to augment real-world training
imagery, especially when CNNs are tested on novel geographic locations or
conditions.

[131]: https://arxiv.org/pdf/2001.05130.pdf

### [ISPRS Vaihingen Dataset][57]

Semantic segmentation for 2D RGB orthophotos of buildings, trees, cars and other
classifications.

Sliding windows with overlap used to derive input images.

9 cm resolution.

IR-R-G

**Determine size of dataset**

### [ISPRS Potsdam Dataset][73]

As with Vaihingen, 2D RGB orthophotos.

5 cm resolution.

**Determine size of dataset**

### [DeepGlobe Dataset][45]

Semantic segmentation for road, building and land cover classification.

Found in spacenet repo. For details:
https://spacenetchallenge.github.io/

also for submission details, log into codalab and navigate to:
https://competitions.codalab.org/competitions/18544#participate

`
aws s3 ls spacenet-dataset/
`

Requested here:
https://competitions.codalab.org/forums/15284/3198/.


### [Dstl Satellite Imagery Feature Detection][44]

Instance segmentation of buildings, roads and 8 other classifications.

Multispectral (3- and 16-band) GeoTiff input images (WorldView 3 sensor).

1km x 1km satellite images.

**resolution and sample size?**

to install:

`
pip install kaggle --upgrade
kaggle competitions download -c dstl-satellite-imagery-feature-detection
`
See:
https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data

Example notebook of mask prediction:
[Full pipeline demo: poly -> pixels -> ML -> poly][74]

### [Massachusetts Buildings Dataset][35]

Semantic segmentation of building/road classes.

51 aerial images of the Boston area, 1500×1500 pixels for an area of 2.25 km²
each, totalling 340 km².

### [The SARptical Dataset][77]

Semantic segmentation on SAR-derived point clouds using optical images.
10k pairs of matched SAR + optical images.

*Not especially relevant for our purposes at the moment, but super cool.*

See also:
[SARptical paper][51]


## Papers


### [CONTEXTUAL PYRAMID ATTENTION NETWORK FOR BUILDING SEGMENTATIONIN AERIAL IMAGERY][159]

Building  extraction  from  aerial  images  has  several  applica-tions in problems such as urban planning, change detection,and disaster management.  With the increasing availability ofdata,  Convolutional  Neural  Networks  (CNNs)  for  semanticsegmentation  of  remote  sensing  imagery  has  improved  sig-nificantly in recent years.  However, convolutions operate inlocal neighborhoods and fail to capture non-local features thatare essential in semantic understanding of aerial images.  Inthis work,  we propose to improve building segmentation ofdifferent  sizes  by  capturing  long-range  dependencies  usingcontextual  pyramid  attention  (CPA).  The  pathways  processthe input at multiple scales efficiently and combine them ina weighted manner, similar to an ensemble model.  The pro-posed method obtains state-of-the-art performance on the In-ria Aerial Image Labelling Dataset with minimal computationcosts.  Our method improves 1.8 points over current state-of-the-art methods and 12.6 points higher than existing baselineson the Intersection over Union (IoU) metric without any post-processing. Code and models will be made publicly available.

[159]: https://arxiv.org/pdf/2004.07018.pdf

### [Adversarial Loss for Semantic Segmentation of Aerial Imagery][158]


Automatic building extraction from aerial imagery has several applications in urban planning, disaster management, and change detection. In recent years, several works have adopted deep convolutional neural networks (CNNs) for building extraction, since they produce rich features that are invariant against lighting conditions, shadows, etc. Although several advances have been made, building extraction from aerial imagery still presents multiple challenges. Most of the deep learning segmentation methods optimize the per-pixel loss with respect to the ground truth without knowledge of the context. This often leads to imperfect outputs that may lead to missing or unrefined regions. In this work, we propose a novel loss function combining both adversarial and cross-entropy losses that learn to understand both local and global contexts for semantic segmentation. The newly proposed loss function deployed on the DeepLab v3+ network obtains state-of-the-art results on the Massachusetts buildings dataset. The loss function improves the structure and refines the edges of buildings without requiring any of the commonly used post-processing methods, such as Conditional Random Fields. We also perform ablation studies to understand the impact of the adversarial loss. Finally, the proposed method achieves a relaxed F1 score of 95.59% on the Massachusetts buildings dataset compared to the previous best F1 of 94.88%. 

[158]: https://arxiv.org/abs/2001.04269


### [Roof material classification from aerial imagery][148]

*useful!*

This paper describes an algorithm for classification ofroof materials using aerial photographs.  Main advantagesof  the  algorithm  are  proposed  methods  to  improve  pre-diction accuracy.   Proposed methods includes:  method ofconverting ImageNet weights of neural networks for usingmulti-channel images; special set of features of second levelmodels that are used in addition to specific predictions ofneural networks;  special set of image augmentations thatimprove training accuracy.  In addition, complete flow forsolving this problem is proposed.  The following content isavailable  in  open  access:  solution  code,  weight  sets  andarchitecture of the used neural networks. The proposed so-lution achieved second place in the competition ”Open AICaribbean Challenge”.

- uses RGB images + IR
- images + roof polygons + metadata + training labels
- useful augmentations
- global properties of buildings in addition to masks
- strategy for extending 3-channel trained network to 4-channels for +IR
- 2 stage - CNN + secondary classifier
- ensemble model (optionally of nets, defo with secondary classifier stage)

[code available](https://github.com/ZFTurbo/DrivenData-Open-AI-Caribbean-Challenge-2nd-place-solution)

[148]: https://arxiv.org/pdf/2004.11482.pdf


### [Hierarchical Multi-Scale Attention for Semantic Segmentation][144]

Multi-scale inference is commonly used to improve the results of semantic segmentation. Multiple images scales are passed through a network and then the results are combined with averaging or max pooling. In this work, we present an attention-based approach to combining multi-scale predictions. We show that predictions at certain scales are better at resolving particular failures modes, and that the network learns to favor those scales for such cases in order to generate better predictions. Our attention mechanism is hierarchical, which enables it to be roughly 4x more memory efficient to train than other recent approaches. In addition to enabling faster training, this allows us to train with larger crop sizes which leads to greater model accuracy. We demonstrate the result of our method on two datasets: Cityscapes and Mapillary Vistas. For Cityscapes, which has a large number of weakly labelled images, we also leverage auto-labelling to improve generalization. Using our approach we achieve a new state-of-the-art results in both Mapillary (61.1 IOU val) and Cityscapes (85.1 IOU test).

85.1 cityscapes

[144]: https://arxiv.org/abs/2005.10821


### [LANet: Local Attention Embedding to Improve the Semantic Segmentation of Remote Sensing Images][145]

**to read**

[145]: https://www.researchgate.net/publication/341685541_LANet_Local_Attention_Embedding_to_Improve_the_Semantic_Segmentation_of_Remote_Sensing_Images

### [Tensor Low-Rank Reconstruction for Semantic Segmentation][143]

Context information plays an indispensable role in the success of semantic segmentation. Recently, non-local self-attention based methods are proved to be effective for context information collection. Since the desired context consists of spatial-wise and channel-wise attentions, 3D representation is an appropriate formulation. However, these non-local methods describe 3D context information based on a 2D similarity matrix, where space compression may lead to channel-wise attention missing. An alternative is to model the contextual information directly without compression. However, this effort confronts a fundamental difficulty, namely the high-rank property of context information. In this paper, we propose a new approach to model the 3D context representations, which not only avoids the space compression but also tackles the high-rank difficulty. Here, inspired by tensor canonical-polyadic decomposition theory (i.e, a high-rank tensor can be expressed as a combination of rank-1 tensors.), we design a low-rank-to-high-rank context reconstruction framework (i.e, RecoNet). Specifically, we first introduce the tensor generation module (TGM), which generates a number of rank-1 tensors to capture fragments of context feature. Then we use these rank-1 tensors to recover the high-rank context features through our proposed tensor reconstruction module (TRM). Extensive experiments show that our method achieves state-of-the-art on various public datasets. Additionally, our proposed method has more than 100 times less computational cost compared with conventional non-local-based methods.

82% cityscapes

[143]: https://arxiv.org/abs/2008.00490

[Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation][142]

Spatial pyramid pooling module or encode-decoder structure are used in deep neural networks for semantic segmentation task. The former networks are able to encode multi-scale contextual information by probing the incoming features with filters or pooling operations at multiple rates and multiple effective fields-of-view, while the latter networks can capture sharper object boundaries by gradually recovering the spatial information. In this work, we propose to combine the advantages from both methods. Specifically, our proposed model, DeepLabv3+, extends DeepLabv3 by adding a simple yet effective decoder module to refine the segmentation results especially along object boundaries. We further explore the Xception model and apply the depthwise separable convolution to both Atrous Spatial Pyramid Pooling and decoder modules, resulting in a faster and stronger encoder-decoder network. We demonstrate the effectiveness of the proposed model on PASCAL VOC 2012 and Cityscapes datasets, achieving the test set performance of 89.0\% and 82.1\% without any post-processing. Our paper is accompanied with a publicly available reference implementation of the proposed models in Tensorflow at \url{this https URL}.

[142] :https://arxiv.org/abs/1802.02611

### [CONTEXTUAL PYRAMID ATTENTION NETWORK FOR BUILDING SEGMENTATIONIN AERIAL IMAGERY][91]

Building extraction from aerial images has several applications in problems such
as urban planning, change detection and disaster management. With the increasing
availability of data, Convolutional Neural Networks (CNNs) for semantic
segmentation of remote sensing imagery has improved significantly in recent
years. However, convolutions operate inlocal neighborhoods and fail to capture
non-local features that are essential in semantic understanding of aerial
images. In this work, we propose to improve building segmentation of different
sizes by capturing long-range dependencies using contextual pyramid attention
(CPA). The pathways process the input at multiple scales efficiently and combine
them in a weighted manner, similar to an ensemble model. The proposed method
obtains state-of-the-art performance on the Inria Aerial Image Labelling Dataset
with minimal computationcosts. Our method improves 1.8 points over current
state-of-the-art methods and 12.6 points higher than existing baselines on the
Intersection over Union (IoU) metric without any post-processing. Code and
models will be made publicly available

** PROMISING! **

[Albumentations][https://github.com/albumentations-team/albumentations] for contrast and brightness augmentation!

### [Generalized Overlap Measures for Evaluation and Validation in Medical Image Analysis][90]

Measures of overlap of labelled regions of images, such as the Dice and Tanimoto
coefficients, have been extensively used to evaluate image registration and
segmentation algorithms. Modern studies can include multiple labels defined on
multiple images yet most evaluation schemes report one overlap per labelled
region, simply averaged over multiple images. In this paper, common overlap
measures are generalized to measure the total overlap of ensembles of labels
defined on multiple test images and account for fractional labels using fuzzy
set theory. This framework allows a single “figure-of-merit” to be reported
which summarises the results of a complex experiment by image pair, by label or
overall. A complementary measure of error, the overlap  distance, is defined
which captures the spatial extent of the nonoverlapping part and is related to
the Hausdorff distance computed on grey level images. The generalized overlap
measures are validated on synthetic images for which the overlap can be computed
analytically and used as similarity measures in nonrigid registration of
three-dimensional magnetic resonance imaging (MRI) brain images. Finally, a
pragmatic segmentation ground truth is constructed by registering a magnetic
resonance atlasbrain to 20 individual scans, and used with the overlap measures
to evaluate publicly available brain segmentation algorithms.

### [Generalised Dice overlap as a deep learning lossfunction for highly unbalanced segmentations][89]

Deep-learning has proved in recent years to be a powerful tool for image
analysis and is now widely used to segment both 2D and 3D medical images.
Deep-learning segmentation frameworks rely not only on the choice of network
architecture but also on the choice of loss function. When the segmentation
process target rare observations, a severe class imbalance is likely to occur
between candidate labels, thus resulting in sub-optimal performance. In order
to mitigate this issue, strategies such as the weighted cross-entropy function,
the sensitivity function or the Dice loss function, have been proposed. In this
work, we investigate the behavior of these loss functions and their sensitivity
to learning rate tuning in the presence of different rates of label imbalance
across 2D and 3D segmentation tasks. We also propose to use the class
re-balancing properties of the Generalized Dice overlap, a known metric for  
segmentation assessment, as a robust and accurate deep-learning loss function
for unbalanced tasks.

### [Automatic Building Extraction in Aerial Scenes Using Convolutional Networks][88]

Automatic building extraction from aerial and satellite imagery is highly
challenging due to extremely large variations of building appearances.
To attack this problem, we design a convolutional network with a final stage
that integrates activations from multiple preceding stages for pixel-wise
prediction, and introduce the signed distance function of building boundaries as
the output representation, which has an enhanced representation power. We
leverage abundant building footprint data available from geographic information
systems (GIS) to compile training data. The trained network achieves superior
performance on datasets that are signihttps://arxiv.org/abs/2008.00490ficantly larger and more complex than
those used in prior work, demonstrating that the proposed method provides a
promising and scalable solution for automating this labor-intensive task.

### [Boundary Loss for Highly Unbalanced Segmentation][87]

**probably very useful!**

Widely used loss functions for convolutional neural network (CNN) segmentation,
e.g., Dice or cross-entropy, are based on integrals (summations) over the
segmentation regions. Unfortunately, for highly unbalanced segmentations, such
regional losses have values that differ considerably – typically of several
orders of magnitude – across segmentation classes, which may affect training
performance and stability. We propose a boundary loss, which takes the form of a
distance metric on the space of contours (or shapes), not regions. This can
mitigate the difficulties of regional losses in the context of highly unbalanced
segmentation problems because it uses integrals over the boundary (interface)
between regions instead of unbalanced integrals over regions. Furthermore, a
boundary loss provides information that is complimentary to regional losses.
Unfortunately, it is not straightforward to represent the boundary points
corresponding to the regional softmax outputs ofa CNN. Our boundary loss is
inspired by discrete (graph-based) optimization techniques for computing
gradient flows of curve evolution.  Following an integral approach for computing
boundary variations, we express a non-symmetric L2 distance on the space of
shapes as a regional integral, which avoids completely local differential
computations involving contour points. This yields a boundary loss expressed
with the regional softmax probability outputs of the network, which can be
easily combined with standard regional losses and implemented with any existing
deep network architecture for N-D segmentation. We report comprehensive
evaluations on two benchmark datasets corresponding to difficult, highly
unbalanced problems: the ischemic stroke lesion (ISLES) and white matter
hyperintensities (WMH). Used in conjunction with the region-based generalized
Dice loss (GDL), our boundary loss improves performance significantly compared
to GDL alone, reaching up to 8% improvement in Dice score and 10% improvement in
Hausdorff score.  It also yielded a more stable learning process. Our code is
publicly available.

[github repository][https://github.com/LIVIAETS/surface-loss]

### [Road Extraction by Deep Residual U-Net][84]

A semantic segmentation neural network which combines the strengths of residual
learning and U-Net is proposed for road area extraction. The network is built
with residual units and has similar architecture to that of U-Net. The benefits
of this model is two-fold: first, residual units ease training of deep networks.
Second, the rich skip connections within the network could facilitate
information propagation, allowing  us to design networks with fewer parameters
however better performance. We test our network on a public road dataset and
compare it with U-Net and other two state of the art deep learning based road
extraction methods. The proposed approach outperforms all the comparing methods,
which demonstrates its superiority over recently developed state of the arts.

*Code available in* [Algorithms](./ALGORITHMS.md)

### [Semantic Segmentation of Urban Buildings from VHR Remote Sensing Imagery Using a Deep CNN (2019)][34]

**Potentially useful!**

A deep convolutional network architecture 'DeepResNet' is presented based on
UNet, which can perform semantic segmentation of urban buildings from VHR
imagery with higher accuracy than other SotA models
(FC/Seg/Deconv/U/ResU/DeepU-nets).

#### Inputs:

The proposed DeepResUnet was tested with aerial images with a spatial
resolution of 0.075 m.

#### Model structure:

The method contains two sub-networks: One is a cascade down-sampling network
for extracting feature maps of buildings from the VHR image, and the other is
an up-sampling network for reconstructing those extracted feature maps back to
the same size of the input VHR image. The deep residual learning approach was
adopted to facilitate training in order to alleviate the degradation problem
that often occurred in the model training process.

#### Results:

**Compared with the U-Net, the F1 score, Kappa coefficient and overall
accuracy of DeepResUnet were improved by 3.52%, 4.67% and 1.72%,
respectively**.

Moreover, the proposed DeepResUnet required much fewer parameters than the
U-Net, highlighting its significant improvement among U-Net applications.

#### Additional observations:

The inference time of DeepResUnet is slightly longer than that of the U-Net.

### [A Relation-Augmented Fully Convolutional Network for Semantic Segmentation in Aerial Scenes (2019)][76]

**Potentially useful**

Abstract:

*Most current semantic segmentation approaches fall back on deep convolutional
neural networks (CNNs). However, their use of convolution operations with local
receptive fields causes failures in modeling contextual spatial relations.
Prior works have sought to address this issue by using graphical models or
spatial propagation modules in networks. But such models often fail to capture
long-range spatial relationships between entities, which leads to spatially
fragmented predictions. Moreover, recent works have demonstrated that
channel-wise information also acts a pivotal part in CNNs. In this work, we
introduce two simple yet effective network units, the spatial relation module
and the channel relation module, to learn and reason about global relationships
between any two spatial positions or feature maps, and then produce
relation-augmented feature representations. The spatial and channel relation
modules are general and extensible, and can be used in a plug-and-play fashion
with the existing fully convolutional network (FCN) framework. We evaluate
relation module-equipped networks on semantic segmentation tasks using two
aerial image datasets, which fundamentally depend on long-range spatial
relational reasoning. The networks achieve very competitive results,
bringing significant improvements over baselines.*

### [Semantic Segmentation of EO Data Using Multi-model and Multi-scale deep networks (2016)][30]

This work investigates the use of deep fully convolutional neural networks
(DFCNN) for pixel-wise scene labelling of EO images. A variant of the SegNet
architecture is trained on remote sensing data over an urban area and different
strategies studied for performing accurate semantic segmentation. Our
contributions are the following:

1. A DFCNN is transferred efficiently from generic everyday images to remote
sensing images
2. A multi-kernel convolutional layer is introduced for fast aggregation of
predictions at multiple scales
3. Data fusion is performed from heterogeneous sensors (optical and laser) using
residual correction. The framework improves state-of-the-art accuracy on the
ISPRS Vaihingen 2D Semantic Labeling dataset.

#### Inputs

Uses the [ISPRS Vaihingen dataset][57].

#### Model Structure

- Initially, SegNet is used (encoder-decoder with VGG-16 pre-trained weights in
  encoder).
- Last layer is parallel multi-kernel (3,5,7) convolutions to aggregate spatial
  information at different scales.


#### Results

*SOTA accuracy, F1 score per-class (2016)*

## Articles

### [An overview of semantic image segmentation][83]

Excellent summary of algorithms and performance improvements gained from
different architectural features.

### [Semantic Segmentation Part 1: DeepLab-V3+][15]

Summarises the DeepLab-V3+ architecture, inference with code examples.

### [Semantic Segmentation Part 2: Training U-Net][16]

Summarises the U-Net architecture, training and inference with code examples.

### [Semantic Segmentation Part 4: State-of-the-Art][14]

Summarises pros and cons of architectures. DeepLab-V3+ for quick feasibility
checks. U-Net for production/small datasets.

### [deepsense.ai Satellite imagery semantic segmentation][9]

Describes application of modified U-Net on
[Dstl Satellite Imagery Feature Detection][44].

--------------------------------------------------------------------------------

# Object Detection

Object detection is a supervised learning problem which aims to predict the
*positions* (specified by bounding boxes) of objects within an image, and the
*probabilities* of each object of belonging to some of a fixed number of
classifications. The number of objects present does not have to be known a priori.

Training data consists of images with object categories and bounding boxes for
each instance of that category.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md).

See also: [Tensorflow Object Detection API][46]

## Datasets

### [xView Dataset][47]

Object detection dataset of 60 classes including buildings.

Satellite imagery, 1m examples in 60 classes, 30cm resolutionm 1415 km^2 area

Comes with pre-trained model.

See also:
[Deep Learning and Satellite Imagery: DIUx Xview Challenge][48]

### [Vehicle Detection in Aerial (VEDAI) Imagery Dataset][56]

Object detection dataset on aerial photographs of vehicles.

1.2k images with 12.5cm/px resolution. RGB and IR channels (separately). Many
angles and illumination conditions present.

9 classes of vehicle labelled, with average of 5.5 per image.

## Papers

### [Segment-before-Detect: Vehicle Detection and Classification through Semantic Segmentation of Aerial Images][25]

This paper uses semantic segmentation followed by connected component analysis
to approximate individual object detection.

### [Speed/accuracy trade-offs for modern convolutional object detectors][1]

### [A Survey on Object Detection in Optical Remote Sensing Images][2]

### [A modified faster R-CNN based on CFAR algorithm for SAR ship detection][59]

### [Learning Rotation-Invariant Convolutional Neural Networks for Object Detection in VHR Optical Remote Sensing Images][60]

### [Automatic Ship Detection of Remote Sensing Images from Google Earth in Complex Scenes Based on Multi-Scale Rotation Dense Feature Pyramid Networks][26]

### As yet unsorted, but potentially useful

[Spectral-spatial classification of hyperspectral imagery with a 3D CNN][27]
[Deep feature extraction and classification of hyperspectral images based on CNNs][28]
[Beyond RGB: Very High Resolution Urban Remote Sensing With Multimodal Deep Networks][29]
[TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation][42]
[Squeeze-and-Excitation Networks][43]
[Residual Hyperspectral Conv-Deconv Network in TensorFlow][52]
[Code for hyperspectral conv-deconv paper][55]
[Building instance classification using street view images][54]
[Technische Universität München Signal Processing in Earth Observation][53]
this looks promising for code examples
[Deep Learning for Spatio temporal Prediction][62]

## Articles

### [Object Detection with Deep Learning on Aerial Imagery][7]
### [Review: SSD — Single Shot Detector (Object Detection)][5]
### [Review: DSSD — Deconvolutional Single Shot Detector (Object Detection)][6]

-------------------------------------------------------------------------------

# Instance Segmentation

Instance Segmentation is a supervised learning problem which goes a step further
by aiming to predict - instead of bounding boxes - *pixel-wise masks* for each
object, along with the associated *classification probabilities*. Hybrid of
semantic segmentation and object detection. The number of objects present does
not have to be known a priori.

*Training data consists of images with object categories and pixel-wise masks
for each instance of that category.*

## Datasets

### [The SpaceNet Off-Nadir Dataset][36]

Instance segmentation of buildings.

Dataset has 120k building footprints over 665 km^2 in Atlanta from 27
associated WV-2 images. Multiple off-nadir angles supplied.

More information can be found at the following links:

`
aws s3 ls spacenet-dataset/
`

[Introducing the SpaceNet Off-Nadir Imagery Dataset][37]
[Challenge results for SpaceNet Off-Nadir Imagery Dataset][38]
[The good and the bad in the SpaceNet Off-Nadir Building Footprint Extraction Challenge][39]
[Winning algorithms for the SpaceNet 4: Off-Nadir Building Footprint Detection Challenge][40]
[The SpaceNet Challenge Off-Nadir Buildings: Introducing the winners][41]

## Papers

## Articles

### [Semantic Segmentation Part 3: Transfer Learning with Mask R-CNN][4]

*TODO*

--------------------------------------------------------------------------------

# Change detection

Change detection is the problem of, given a series of co-registered images
displaced in time, identifying areas that have changed. This more complex than
analysing a simple difference of images, as some variable processes such as
illumination and weather effects are not considered true changes and algorithms
must learn to ignore these.

Supervised change detection algorithms typically use pairs of before and after
images, along with a mask identifying which pixels are considered to have changed.
More complex approaches might additionally distinguish the semantics of changes.

Current methods typically factorise into two kinds:

- *Post-classification comparison*: First classify the content of two temporally
different images of the same scene then compare to identify differences.
Inaccuracies arise due to errors in classification in either of the two images,
so an accurate classifier is required.

- *Difference image analysis*: Construct a DI to highlight differences. Further
analysis is then performed to determine nature of changes. CD results depend on
quality of produced DI. Atmospheric effects on reflectance values necessitate
techniques like radiometric correction, spectral differencing, rationing and
texture rationing.

For more information on the machine-learning techniques used, see
[Algorithms](./ALGORITHMS.md). **Actually, look below at papers atm.**

## Datasets

### [Onera Satellite Change Detection Dataset][32]
Pairs of Sentinel-2 multispectral images of urban areas, with pixelwise
*artificialisation, ie urban only* changes annotated as binary(no-)change).
10-60m resolution with 30 bands.

## Papers

### Common themes

A common theme in the literature is that deep-learning-based change detection
algorithms are not yet efficient and are limited by their training datasets.

Another is that object-based approaches to change detection (those which
segment the image's objects then compare them) are preferable to pixel- or
kernel-based methods, since the latter compare an insufficient amount of
information and cause irregular boundaries.

Yet another is that transfer-learning with networks pre-trained on large datasets
like ImageNet works successfully for CD architectures and is useful due to a
lack of remote-sensing training data.

weakly-supervised

125-130

### [Transfer Change Rules from Recurrent FullyConvolutional Networks for HyperspectralUnmanned Aerial Vehicle Images without GroundTruth Data][157]

Change detection (CD) networks based on supervised learning have been used in diverseCD tasks.  However, such supervised CD networks require a large amount of data and only useinformation from current images. In addition, it is time consuming to manually acquire the groundtruth data for newly obtained images. Here, we proposed a novel method for CD in case of a lack oftraining data in an area near by another one with the available ground truth data. The proposed methodautomatically entails generating training data and fine-tuning the CD network. To detect changesin target images without ground truth data, the difference images were generated using spectralsimilarity measure, and the training data were selected via fuzzy c-means clustering.  Recurrentfully convolutional networks with multiscale three-dimensional filters were used to extract objectsof various sizes from unmanned aerial vehicle (UAV) images. The CD network was pre-trained onlabeled source domain data; then, the network was fine-tuned on target images using generatedtraining data.   Two further CD networks were trained with a combined weighted loss function.The training data in the target domain were iteratively updated using he prediction map of the CDnetwork. Experiments on two hyperspectral UAV datasets confirmed that the proposed method iscapable of transferring change rules and improving CD results based on training data extracted in anunsupervised way.

[157]: https://www.researchgate.net/publication/340344151_Transfer_Change_Rules_from_Recurrent_Fully_Convolutional_Networks_for_Hyperspectral_Unmanned_Aerial_Vehicle_Images_without_Ground_Truth_Data/link/5e886347a6fdcca789f190b7/download

### [Change Detection on Multi-Spectral Images Based on Feature-level U-Net][156]

This paper proposes a change detection algorithm on multi-spectral images based on feature-level U-Net. A low-complexity pan-sharpening method is proposed to employ not only panchromatic images, but also multi-spectral images for enhancing the performance of the deep neural network. The high-resolution multi-spectral (HRMS) images are then fed into the proposed feature-level U-Net. The proposed feature-level U-Net consists of two-stages: a feature-level subtracting network and U-Net. The feature-level subtracting network is used to extract dynamic difference images (DI) for the use of low-level and high-level features. By employing this network, the performance of change detection algorithms can be improved with a smaller number of layers for U-Net with a low computational complexity. Furthermore, the proposed algorithm detects small changes by taking benefits of both geometrical and spectral resolution enhancement and adopting an intensity-hue-saturation (IHS) pan-sharpening method. A modified of IHS pan-sharpening algorithm is introduced to solve spectral distortion problem by applying mean filtering in high frequency. We found that the proposed change detection on HRMS images gives a better performance compared to existing change detection algorithms by achieving an average F-1 score of 0.62, a percentage correct classification (PCC) of 98.78%, and a kappa of 61.60 for test datasets.

[156]: https://ieeexplore.ieee.org/abstract/document/8952681


### [Fully Convolutional Networks with Multiscale 3D Filters and Transfer Learning for Change Detection in High Spatial Resolution Satellite Images][155]

- uses trained semantic segmentation model backbone into recurrent FCN for change part
- fully supervised

Remote sensing images having high spatial resolution are acquired, and large amounts of data are extracted from their region of interest. For processing these images, objects of various sizes, from very small neighborhoods to large regions composed of thousands of pixels, should be considered. To this end, this study proposes change detection method using transfer learning and recurrent fully convolutional networks with multiscale three-dimensional (3D) filters. The initial convolutional layer of the change detection network with multiscale 3D filters was designed to extract spatial and spectral features of materials having different sizes; the layer exploits pre-trained weights and biases of semantic segmentation network trained on an open benchmark dataset. The 3D filter sizes were defined in a specialized way to extract spatial and spectral information, and the optimal size of the filter was determined using highly accurate semantic segmentation results. To demonstrate the effectiveness of the proposed method, binary change detection was performed on images obtained from multi-temporal Korea multipurpose satellite-3A. Results revealed that the proposed method outperformed the traditional deep learning-based change detection methods and the change detection accuracy improved using multiscale 3D filters and transfer learning.

[155]: https://www.mdpi.com/2072-4292/12/5/799/htm

### [An Efficient Change Detection for Large SARImages Based on Modified U-Net Framework][154]

- ~kinda supervised binary change-no-change
- training samples semi-automatically generated by DI + morphological cleaning + manual correction
- modified U-net

Large SAR images usually contain a variety of land-cover types and accordingly complicatedchange types, which cause great difficulty for accurate change detection. The U-Net is aspecial fully convolutional neural network that not only can capture multiple features in theimage context but also enables precise pixel-by-pixel image classification. Therefore, weexplore the U-Net to describe accurately the differences between bi-temporal SAR imagesfor high-precision change detection. However, large scene SAR images often have signifi-cantly different statistical distributions for various change types, which prevents the U-Netfrom working properly. We modified the U-Net by introducing the batch normalization (BN)operation at the input of every neuron to regularize the statistical distributions of its inputdata for avoiding the risk of gradient disappearance or dispersion during the network train-ing. In addition, the ELU (Exponential Linear Unit) activation function replaces the ReLU(Rectified Linear Unit) function to improve further the gradients backpropagation. Then weselected bi-temporal Sentinel-1SAR data covering Jiangsu Province, China, to discuss quanti-tatively and qualitatively the detection performance and model complexity of the modifiednetwork with different numbers of convolutional kernels.

[154]: https://sci-hub.tw/10.1080/07038992.2020.1783993

### [DSDANet: Deep Siamese Domain Adaptation Convolutional Neural Network for Cross-domain Change Detection][153]

*focuses on assimilation of different, domain-specific CD networks*

Change detection (CD) is one of the most vital applications in remote sensing. Recently, deep learning has achieved promising performance in the CD task. However, the deep models are task-specific and CD data set bias often exists, hence it is inevitable that deep CD models would suffer degraded performance after transferring it from original CD data set to new ones, making manually label numerous samples in the new data set unavoidable, which costs a large amount of time and human labor. How to learn a transferable CD model in the data set with enough labeled data (original domain) but can well detect changes in another data set without labeled data (target domain)? This is defined as the cross-domain change detection problem. In this paper, we propose a novel deep siamese domain adaptation convolutional neural network (DSDANet) architecture for cross-domain CD. In DSDANet, a siamese convolutional neural network first extracts spatial-spectral features from multi-temporal images. Then, through multi-kernel maximum mean discrepancy (MK-MMD), the learned feature representation is embedded into a reproducing kernel Hilbert space (RKHS), in which the distribution of two domains can be explicitly matched. By optimizing the network parameters and kernel coefficients with the source labeled data and target unlabeled data, DSDANet can learn transferrable feature representation that can bridge the discrepancy between two domains. To the best of our knowledge, it is the first time that such a domain adaptation-based deep network is proposed for CD. The theoretical analysis and experimental results demonstrate the effectiveness and potential of the proposed method.

[153]: https://arxiv.org/abs/2006.09225

### [A Survey of Change Detection Methods Based on Remote Sensing Images for Multi-Source and Multi-Objective Scenarios][152]

- overview of architectures and algorithms
- good summary of co-registration/preprocessing strategies
- good summary of datasets for SAR/VHR/hyperspectral
- discussion of roads vs buildings

Quantities of multi-temporal remote sensing (RS) images create favorable conditions for exploring the urban change in the long term. However, diverse multi-source features and change patterns bring challenges to the change detection in urban cases. In order to sort out the development venation of urban change detection, we make an observation of the literatures on change detection in the last five years, which focuses on the disparate multi-source RS images and multi-objective scenarios determined according to scene category. Based on the survey, a general change detection framework, including change information extraction, data fusion, and analysis of multi-objective scenarios modules, is summarized. Owing to the attributes of input RS images affect the technical selection of each module, data characteristics and application domains across different categories of RS images are discussed firstly. On this basis, not only the evolution process and relationship of the representative solutions are elaborated in the module description, through emphasizing the feasibility of fusing diverse data and the manifold application scenarios, we also advocate a complete change detection pipeline. At the end of the paper, we conclude the current development situation and put forward possible research direction of urban change detection, in the hope of providing insights to the following research.

[152]: https://www.mdpi.com/2072-4292/12/15/2460


### [Unsupervised Change Detection in Multi-temporal VHR Images Based on Deep Kernel PCA Convolutional Mapping Network (2020)][151]

*interesting. not DL. multiclass and unsupervised.*

- image registration and radiometric (z-standarisation) correction etc
- multiple coregistered patch-pairs selected and used to train KPCA kernels (shared weights between input images)
- feature maps generated by KPCA kernels
- patch-selection and training of subsequent layer KPCA filters and repeat feature map gen
- pixelwise subtraction gives pixel difference map
- magnitude thresholding gives binary change map
- define "type" of change by via an angle: the ratio of the eigenvalue-weighted sum of
difference maps from last layer PCA kernel to the sum of squares of these over all diffrence
maps

[151]: https://arxiv.org/pdf/1912.08628.pdf

### [Automatic Change Detection in Synthetic Aperture Radar Images Based on PCANet (2016)][67]

- log-ratio image generated
- gabor wavelets + fuzzy c-means used on this for preselection of regions around likely changed pixels
- PCA trained on these patches
- some L1 principal eigenvectors chosen as PCA filters
- repeat for however many layers
- PCA model used to classify remaining regions as changed/unchanged classes
  - binarise output of filters using step function 1 for positive input else 0
  - around each pixel vector of L2 binary bits converted to hist
  - Linear SVM classifier on hist
- this classification result and preclassification result combined into final change maps

*not done on GPUs*

This letter presents a novel change detection methodfor  multitemporal  synthetic  aperture  radar  images  based  onPCANet. This method exploits representative neighborhood fea-tures  from  each  pixel  using  PCA  filters  as  convolutional  filters.Thus,  the  proposed  method  is  more  robust  to  the  speckle  noiseand  can  generate  change  maps  with  less  noise  spots.  Given  twomultitemporal  images,  Gabor  wavelets  and  fuzzyc-means  areutilized  to  select  interested  pixels  that  have  high  probability  ofbeing changed or unchanged. Then, new image patches centeredat   interested   pixels   are   generated   and   a   PCANet   model   istrained using these patches. Finally, pixels  in the multitemporalimages are classified by the trained PCANet model. The PCANetclassification result and the preclassification result are combinedto form the final change map. The experimental results obtainedon  three  real  SAR  image  data  sets  confirm  the  effectiveness  ofthe proposed method.

### [A Deep Learning Method for Change Detection in Synthetic Aperture Radar Images (2019)][150]

- generate pixel-wise similarity matrix (|diff|/sum)
- spatial fuzzy clustering algo: iterative partition of N pixels into c fuzzy sets.
c set to 2 (C/NC). m=2 weighting exponent. membership then modified by local neighbourhood
spatial sterm to combat speckle noise. => binary false label matrix.
- selected training samples chosen and basic 2-layer CNN classifier trained

*binary*

With the rapid development of various technologies of satellite sensor, synthetic aperture radar (SAR) image has been an import source of data in the application of change detection. In this paper, a novel method based on a convolutional neural network (CNN) for SAR image change detection is proposed. The main idea of our method is to generate the classification results directly from the original two SAR images through a CNN without any preprocessing operations, which also eliminate the process of generating the difference image (DI), thus reducing the influence of the DI on the final classification result. In CNN, the spatial characteristics of the raw image can be extracted and captured by automatic learning and the results with stronger robustness can be obtained. The basic idea of the proposed method includes three steps: it first produces false labels through unsupervised spatial fuzzy clustering. Then we train the CNN through proper samples that are selected from the samples with false labels. Finally, the final detection results are obtained by the trained convolutional network. Although training the convolutional network is a supervised learning fashion, the whole process of the algorithm is an unsupervised process without priori knowledge. The theoretical analysis and experimental results demonstrate the validity, robustness, and potential of our algorithm in simulated and real data sets. In addition, we try to apply our algorithm to the change detection of heterogeneous images, which also achieves satisfactory results.

[150]: https://sci-hub.tw/10.1109/tgrs.2019.2901945

### [A Convolutional Neural Network with Parallel Multi-Scale Spatial Pooling to Detect Temporal Changes in SAR Images (2020)][149]

*useful - can improve on PCAnet style architectures with spatial pooling*
*binary change/no-change*
*fully supervised, manually annotated change maps*

In synthetic aperture radar (SAR) image change detection, it is quite challenging to exploit the changing information from the noisy difference image subject to the speckle. In this paper, we propose a multi-scale spatial pooling (MSSP) network to exploit the changed information from the noisy difference image. Being different from the traditional convolutional network with only mono-scale pooling kernels, in the proposed method, multi-scale pooling kernels are equipped in a convolutional network to exploit the spatial context information on changed regions from the difference image. Furthermore, to verify the generalization of the proposed method, we apply our proposed method to the cross-dataset bitemporal SAR image change detection, where the MSSP network (MSSP-Net) is trained on a dataset and then applied to an unknown testing dataset. We compare the proposed method with other state-of-arts and the comparisons are performed on four challenging datasets of bitemporal SAR images. Experimental results demonstrate that our proposed method obtains comparable results with S-PCA-Net on YR-A and YR-B dataset and outperforms other state-of-art methods, especially on the Sendai-A and Sendai-B datasets with more complex scenes. More important, MSSP-Net is more efficient than S-PCA-Net and convolutional neural networks (CNN) with less executing time in both training and testing phases.

[149]: https://arxiv.org/abs/2005.10986

### [A Framework for Automatic and Unsupervised Detection of Multiple Changes in Multitemporal Images][147]

*C2VA*

The  detection  of  multiple  changes  (i.e.,  differentkinds  of  change)  in  multitemporal  remote  sensing  images  is  acomplex problem. When multispectral images havingBspectralbands  are  considered,  an  effective  solution  to  this  problem  isto  exploit  all  available  spectral  channels  in  the  framework  ofsupervised or partially supervised approaches. However, in manyreal applications, it is difficult/impossible to collect ground truthinformation for either multitemporal or single-date images. On theopposite, unsupervised methods available in the literature are noteffective in handling the full information present in multispectraland  multitemporal  images.  They  usually  consider  a  simplifiedsubspace of the original feature space having small dimensionalityand, thus, characterized by a possible loss of change information.In this paper, we present a framework for the detection of multiplechanges  in  bitemporal  and  multispectral  remote  sensing  imagesthat allows one to overcome the limits of standard unsupervisedmethods.  The  framework  is  based  on  the  following:  1)  a  com-pressed yet efficient 2-D representation of the change informationand  2)  a  two-step  automatic  decision  strategy.  The  effectivenessof the proposed approach has been tested on two bitemporal andmultispectral  data  sets  having  different  properties.  Results  ob-tained on both data sets confirm the effectiveness of the proposedapproach.

[147]: https://sci-hub.se/10.1109/tgrs.2011.2171493

### [Multiscale Morphological Compressed ChangeVector Analysis for Unsupervised MultipleChange Detection][146]

A novel multiscale morphological compressed changevector analysis (M2C2VA)
method is proposed to address the multiple-change detection problem (i.e.
identifying different classes of changes) in bitemporal remote sensing images.
The proposed approach contributes to extend the state-of-the-art spectrum-based
compressed change vector analysis (C2VA) method by jointly analyzing the
spectral-spatial change informa-tion. In greater details, reconstructed spectral change vector fea-tures are built according to a morphological analysis. Thus moregeometrical details of change classes are preserved while exploitingthe interaction of a pixel with its adjacent regions. Two multiscaleensemble strategies, i.e., data level and decision level fusion, aredesigned to integrate the change information represented at dif-ferent scales of features or to combine the change detection resultsobtained by the detector at different scales, respectively. A detailedscale sensitivity analysis is carried out to investigate its impacts onthe performance of the proposed method. The proposed method isdesigned in an unsupervised fashion without requiring any groundreference data. The proposed M2C2VA is tested on one simulatedand three real bitemporal remote sensing images showing its prop-erties in terms of different image size and spatial resolution. Ex-perimental results confirm its effectiveness.

[146]: https://sci-hub.tw/10.1109/jstars.2017.2712119

### [A Light-Weighted Convolutional Neural Network for BitemporalSAR Image Change Detection][123]

Recently, many Convolution Neural Networks (CNN) have been successfully employed
in bitemporal SAR image change detection. However, most of the existing networks
are too heavy and occupy a large volume of memoryfor storage and calculation.
Motivated by this, in this paper, we propose a lightweight neural network to
reduce the computational and spatial complexity and facilitate the change
detection on an edge device. In the proposed network, we replace normal
convolutional layers with bottleneck layers that keep the same number of
channels between input and output. Next, we employ dilated convolutional kernels
with a few non-zero entries that reduce the running time in convolutional
operators. Comparing with the conventional convolutional neural network, our
light-weighted neural network will be more efficient with fewer parameters.
We verify our light-weighted neural network on four sets of bitemporal SAR
images. The experimental results show that the proposed network can obtain
better performance than the conventional CNN and has better model
generalisation, especially on the challenging datasets with complex scenes

[123]: https://arxiv.org/pdf/2005.14376.pdf

### [S2-cGAN: Self-Supervised Adversarial Representation Learning for Binary Change Detection in Multispectral Images (2020)][122]

Deep Neural Networks have recently demonstrated promising performance in binary
change detection (CD) problems in remote sensing (RS), requiring a large amount
of labeled multitemporal training samples. Since collecting such data is
time-consuming and costly, most of the existing methods rely on pre-trained
networks on publicly available computer vision (CV) datasets. However, because
of the differences in image characteristics in CV and RS, this approach limits
the performance of the existing CD methods. To address this problem, we propose
a self-supervised conditional Generative Adversarial Network (S2-cGAN). The
proposed S^2-cGAN is trained to generate only the distribution of unchanged
samples. To this end, the proposed method consists of two main steps: 1)
Generating a reconstructed version of the input image as an unchanged image 2)
Learning the distribution of unchanged samples through an adversarial game.
Unlike the existing GAN based methods (which only use the discriminator during
the adversarial training to supervise the generator), the S2-cGAN directly
exploits the discriminator likelihood to solve the binary CD task. Experimental
results show the effectiveness of the proposed S2-cGAN when compared to the
state of the art CD methods.

[122]: https://arxiv.org/abs/2007.02565


### [Feature learning and change feature classification based on deeplearning for ternary change detection in SAR images][132]


**useful!**


Ternary change detection aims to detect changes and group the changes into positive change and nega-tive change. It is of great significance in the joint interpretation of spatial-temporal synthetic apertureradar images. In this study, sparse autoencoder, convolutional neural networks (CNN) and unsupervisedclustering are combined to solve ternary change detection problem without any supervison. Firstly,sparse autoencoder is used to transform log-ratio difference image into a suitable feature space forextracting key changes and suppressing outliers and noise. And then the learned features are clusteredinto three classes, which are taken as the pseudo labels for training a CNN model as change feature clas-sifier. The reliable training samples for CNN are selected from the feature maps learned by sparse autoen-coder with certain selection rules. Having training samples and the corresponding pseudo labels, the CNNmodel can be trained by using back propagation with stochastic gradient descent. During its training pro-cedure, CNN is driven to learn the concept of change, and more powerful model is established to distin-guish different types of changes. Unlike the traditional methods, the proposed framework integrates themerits of sparse autoencoder and CNN to learn more robust difference representations and the concept ofchange for ternary change detection. Experimental results on real datasets validate the effectiveness andsuperiority of the proposed framework.

[132]:https://sci-hub.tw/10.1016/j.isprsjprs.2017.05.001

Some fully unsupervised CNN-based methods have achieved positive results. In [132]

### [A new difference image creation method based on deep neural networks for change detection in remote-sensing images (2017)][133]

In this article, we propose a novel difference image (DI) creationmethod for unsupervised change detection in multi-temporal multi-spectral remote-sensing images based on deep learning theory.First, we apply deep belief network to learn local and high-levelfeatures from the local neighbour of a given pixel in an unsuper-vised manner. Second, a back propagation algorithm is improved tobuild a DI based on selected training samples, which can highlightthe difference on changed regions and suppress the false changeson unchanged regions. Finally, we get the change trajectory mapusing simple clustering analysis. The proposed scheme is tested onthree remote-sensing data sets. Qualitative and quantitative evalua-tions show its superior performance compared to the traditionalpixel-level and texture-level-based approaches.

[133]: https://sci-hub.tw/10.1080/01431161.2017.1371861


### [Unsupervised DifferenceRepresentation Learningfor Detecting Multiple Types of Changes in Multitemporal Remote Sensing Images (2018)][134]

**very useful!**

With the rapid increase of remote sensing images intemporal, spectral, and spatial resolutions, it is urgent to developeffective  techniques  for  joint  interpretation  of  spatial-temporalimages. Multitype change detection (CD) is a significant researchtopic  in  multitemporal  remote  sensing  image  analysis,  and  itscore is to effectively measure the difference degree and representthe  difference  among  the  multitemporal  images.  In  this  paper,we  propose  a  novel  difference  representation  learning  (DRL)network  and  present  an  unsupervised  learning  framework  formultitype  CD  task.  Deep  neural  networks  work  well  in  rep-resentation  learning  but  rely  too  much  on  labeled  data,  whileclustering  is  a  widely  used  classification  technique  free  fromsupervision. However, the distribution of real remote sensing datais  often  not  very  friendly  for  clustering.  To  better  highlight  thechanges  and  distinguish different types  of  changes,  we  combinedifference measurement, DRL, and unsupervised clustering into aunified model, which can be driven to learn Gaussian-distributedand discriminative difference representations for nonchange anddifferent  types  of  changes.  Furthermore,  the  proposed  model  isextended  into  an  iterative  framework  to  imitate  the  bottom-upaggregative  clustering procedure, in  which similar  change  typesare  gradually  merged  into  the  same  classes.  At  the  same  time,the  training  samples  are  updated  and  reused  to  ensure  that  itconverges  to a stable solution. The experimental  studies on fourpairs of multispectral data sets demonstrate the effectiveness andsuperiority of the proposed model on  multitype CD

multitype CD  methods [5], [16].  BinaryCD focuses on highlighting the changes, while multitype CDaims to further distinguish different types of changes. In thispaper, we focus on the multitype CD problem. Change vectoranalysis  (CVA) [17]  is  a  classical  multitype  CD  techniquewhich  has  been  fully  studiedin  recent  years.  Some  vari-ants  of  CVA  also  have  been  proposed,  such  as  compressedCVA  (C2VA) [18],  sequential  spectral  CVA    [3],  improvedC2VA [19], and multiscale morphological C2VA  [ 5 ]  e

[134]: https://sci-hub.tw/10.1109/tgrs.2018.2872509

### [Differentially Deep Subspace Representation for Unsupervised Change Detection of SAR Images][135]

*binary change-no-change*

Temporal analysis of synthetic aperture radar (SAR) time series is a basic and significantissue in the remote sensing field. Change detection as well as other interpretation tasks of SAR imagesalways involves non-linear/non-convex problems. Complex (non-linear) change criteria or modelshave thus been proposed for SAR images, instead of direct difference (e.g., change vector analysis)with/without linear transform (e.g., Principal Component Analysis, Slow Feature Analysis) used inoptical image change detection. In this paper, inspired by the powerful deep learning techniques, wepresent a deep autoencoder (AE) based non-linear subspace representation for unsupervised changedetection with multi-temporal SAR images. The proposed architecture is built upon an autoencoder-like(AE-like) network, which non-linearly maps the input SAR data into a latent space. Unlike normal AEnetworks, a self-expressive layer performing like principal component analysis (PCA) is added betweenthe encoder and the decoder, which further transforms the mapped SAR data to mutually orthogonalsubspaces. To make the proposed architecture more efficient at change detection tasks, the parametersare trained to minimize the representation difference of unchanged pixels in the deep subspace. Thus,the proposed architecture is namely the Differentially Deep Subspace Representation (DDSR) networkfor multi-temporal SAR images change detection.Experimental results on real datasets validate theeffectiveness and superiority of the proposed architecture.

[135]: https://www.researchgate.net/publication/337449331_Differentially_Deep_Subspace_Representation_for_Unsupervised_Change_Detection_of_SAR_Images


### [DSDANet: Deep Siamese Domain Adaptation Convolutional Neural Network for Cross-domain Change Detection][139]

*to read*

Change detection (CD) is one of the most vital applications in remote sensing. Recently, deep learning has achieved promising performance in the CD task. However, the deep models are task-specific and CD data set bias often exists, hence it is inevitable that deep CD models would suffer degraded performance after transferring it from original CD data set to new ones, making manually label numerous samples in the new data set unavoidable, which costs a large amount of time and human labor. How to learn a transferable CD model in the data set with enough labeled data (original domain) but can well detect changes in another data set without labeled data (target domain)? This is defined as the cross-domain change detection problem. In this paper, we propose a novel deep siamese domain adaptation convolutional neural network (DSDANet) architecture for cross-domain CD. In DSDANet, a siamese convolutional neural network first extracts spatial-spectral features from multi-temporal images. Then, through multi-kernel maximum mean discrepancy (MK-MMD), the learned feature representation is embedded into a reproducing kernel Hilbert space (RKHS), in which the distribution of two domains can be explicitly matched. By optimizing the network parameters and kernel coefficients with the source labeled data and target unlabeled data, DSDANet can learn transferrable feature representation that can bridge the discrepancy between two domains. To the best of our knowledge, it is the first time that such a domain adaptation-based deep network is proposed for CD. The theoretical analysis and experimental results demonstrate the effectiveness and potential of the proposed method.

[139]: https://www.researchgate.net/publication/342229780_DSDANet_Deep_Siamese_Domain_Adaptation_Convolutional_Neural_Network_for_Cross-domain_Change_Detection


### [Change Detection in Multisource VHR Images via Deep Siamese Convolutional Multiple-Layers Recurrent Neural Network (2019)][140]

*deep siamese CNN to extract sequences of spectral/spatial features -> stacked LSTMs
extract spectral/spatial/temporal features from these -> DNN classifier. weird explanation for why no maxpooling "because images are multitemporal". ambiguously
defined FC stage acts on 'pixel' index of spatiotemporal features, which apparently is
integrated out by a W x W convolution beforehand...*

With the rapid development of Earth observation technology, very-high-resolution (VHR) images from various satellite sensors are more available, which greatly enrich the data source of change detection (CD). Multisource multitemporal images can provide abundant information on observed landscapes with various physical and material views, and it is exigent to develop efficient techniques to utilize these multisource data for CD. In this article, we propose a novel and general deep siamese convolutional multiple-layers recurrent neural network (RNN) (SiamCRNN) for CD in multitemporal VHR images. Superior to most VHR image CD methods, SiamCRNN can be used for both homogeneous and heterogeneous images. Integrating the merits of both convolutional neural network (CNN) and RNN, SiamCRNN consists of three subnetworks: deep siamese convolutional neural network (DSCNN), multiple-layers RNN (MRNN), and fully connected (FC) layers. The DSCNN has a flexible structure for multisource image and is able to extract spatial-spectral features from homogeneous or heterogeneous VHR image patches. The MRNN stacked by long-short term memory (LSTM) units is responsible for mapping the spatial-spectral features extracted by DSCNN into a new latent feature space and mining the change information between them. In addition, FC, the last part of SiamCRNN, is adopted to predict change probability. The experimental results in two homogeneous data sets and one challenging heterogeneous VHR images data set demonstrate that the promising performances of the proposed network outperform several state-of-the-art approaches.

[140]: https://www.researchgate.net/publication/338144403_Change_Detection_in_Multisource_VHR_Images_via_Deep_Siamese_Convolutional_Multiple-Layers_Recurrent_Neural_Network


### [Unsupervised Change Detection in Multi-temporal VHR Images Based on Deep Kernel PCA Convolutional Mapping Network (2019)][136]

**useful - unsupervised multiclass change det. bitemporal. KPCA conv kernels. no code.**

 Lyuet al [44] adopt recurrent neural network (RNN) to tackle thetemporal  connection  between  multi-temporal  images.  Goingone step further, in [45], a CD architecture based on recurrentconvolutional neural network is proposed to extract unified fea-tures for binary and multi-class CD. Combining a pre-traineddeep convolutional neural network with CVA, Saha et al [46]design  a  CD  method  called  deep  CVA  for  VHR  images  CD.In [47], Daudt et al first introduce fully convolutional network(FCN) into CD and propose two siamese extensions of FCN,which achieve good performance in two open VHR images CDdatasets.

 Learning  nonlinear  features  with  DNNand  highlighting  changes  via  SFA,  Ru  et  al  [52]  proposean  unsupervised  deep  slow  feature  analysis  (DSFA)  modelfor  CD.  For  training  the  DNN  of  DSFA,  a  pre-detectionmethod based on CVA is utilized to selecting samples. In [53],a  supervised  spatial  fuzzy  clustering  is  adopted  to  producepseudo-labels  for  training  the  DCNN.  This  approach  solvesthe  sample  problem  of  the  DL  model  to  a  certain  degree.However, if the pre-detection algorithm does not perform wellon one data set, the performance of DL model is also damaged.Whats  more,  most  of  these  existing  DL-based  methods  aremerely focus on binary CD. And there are currently only a fewmethods  [46],  [54]  that  can  be  used  for  unsupervised  multi-class CD.

With the development of Earth observation technology, very-high-resolution (VHR) image has become an important data source of change detection. Nowadays, deep learning methods have achieved conspicuous performance in the change detection of VHR images. Nonetheless, most of the existing change detection models based on deep learning require annotated training samples. In this paper, a novel unsupervised model called kernel principal component analysis (KPCA) convolution is proposed for extracting representative features from multi-temporal VHR images. Based on the KPCA convolution, an unsupervised deep siamese KPCA convolutional mapping network (KPCA-MNet) is designed for binary and multi-class change detection. In the KPCA-MNet, the high-level spatial-spectral feature maps are extracted by a deep siamese network consisting of weight-shared PCA convolution layers. Then, the change information in the feature difference map is mapped into a 2-D polar domain. Finally, the change detection results are generated by threshold segmentation and clustering algorithms. All procedures of KPCA-MNet does not require labeled data. The theoretical analysis and experimental results demonstrate the validity, robustness, and potential of the proposed method in two binary change detection data sets and one multi-class change detection data set.

[136]: https://www.researchgate.net/publication/338033327_Unsupervised_Change_Detection_in_Multi-temporal_VHR_Images_Based_on_Deep_Kernel_PCA_Convolutional_Mapping_Network


### [Toward Generalized Change Detection on Planetary Surfaces With Convolutional Autoencoders and Transfer Learning (2019)][137]

*to read*

Ongoing planetary exploration missions are returning large volumes of image data. Identifying surface changes in these images, e.g., new impact craters, is critical for investigating many scientific hypotheses. Traditional approaches to change detection rely on image differencing and manual feature engineering. These methods can be sensitive to irrelevant variations in illumination or image quality and typically require before and after images to be coregistered, which itself is a major challenge. Additionally, most prior change detection studies have been limited to remote sensing images of earth. We propose a new deep learning approach for binary patch-level change detection involving transfer learning and nonlinear dimensionality reduction using convolutional autoencoders. Our experiments on diverse remote sensing datasets of Mars, the moon, and earth show that our methods can detect meaningful changes with high accuracy using a relatively small training dataset despite significant differences in illumination, image quality, imaging sensors, coregistration, and surface properties. We show that the latent representations learned by a convolutional autoencoder yield the most general representations for detecting change across surface feature types, scales, sensors, and planetary bodies.

[137]:  https://www.researchgate.net/publication/335722445_Toward_Generalized_Change_Detection_on_Planetary_Surfaces_With_Convolutional_Autoencoders_and_Transfer_Learning


### [End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++ (2019)][138]

*Looks like the way to go for supervised CD, using existing deepresunet*

Change detection (CD) is essential to the accurate understanding of land surface changes using available Earth observation data. Due to the great advantages in deep feature representation and nonlinear problem modeling, deep learning is becoming increasingly popular to solve CD tasks in remote-sensing community. However, most existing deep learning-based CD methods are implemented by either generating difference images using deep features or learning change relations between pixel patches, which leads to error accumulation problems since many intermediate processing steps are needed to obtain final change maps. To address the above-mentioned issues, a novel end-to-end CD method is proposed based on an effective encoder-decoder architecture for semantic segmentation named UNet++, where change maps could be learned from scratch using available annotated datasets. Firstly, co-registered image pairs are concatenated as an input for the improved UNet++ network, where both global and fine-grained information can be utilized to generate feature maps with high spatial accuracy. Then, the fusion strategy of multiple side outputs is adopted to combine change maps from different semantic levels, thereby generating a final change map with high accuracy. The effectiveness and reliability of our proposed CD method are verified on very-high-resolution (VHR) satellite image datasets. Extensive experimental results have shown that our proposed approach outperforms the other state-of-the-art CD methods.

[138]: https://www.mdpi.com/2072-4292/11/11/1382/htm

### [Change Detection Based on Artificial Intelligence: State-of-the-Art and Challenges (2020)][121]

**very useful - especially sections 4.3 and 4.4**

unsupervised, latent change map: refs [107,157,163,184]

kernel PCA convolution to extract representative spatial–spectral features from RS images (unsupervised)[163]

urban context change detection refs [229, 52, 228, 112, 177, 85, 192]


In [94], the authors proposed a pyramid of feature-based attention-guided
Siamese network to detect building changes and the IoU of changemap exceeded 0.97.

Abstract:

Change detection based on remote sensing (RS) data is an important method of
detecting changes on the Earth’s surface and has a wide range of applications in
urban planning, environmental monitoring, agriculture investigation, disaster
assessment, and map revision. In recent years, integrated artificial
intelligence (AI) technology has become a research focus in developing new
change detection methods. Although some researchers claim that AI-based change
detection approaches outperform traditional change detection approaches, it is
not immediately obvious howand to what extent AI can improve the performance of
change detection.  This review focuses onthe state-of-the-art methods,
applications, and challenges of AI for change detection. Specifically, the
implementation process of AI-based change detection is first introduced. Then,
the data from different sensors used for change detection, including optical RS
data, synthetic aperture radar (SAR) data, street view images, and combined
heterogeneous data, are presented, and the available open datasets are also
listed. The general frameworks of AI-based change detection methods are reviewed
and analyzed systematically, and the unsupervised schemes used in AI-based change detectionare further analyzed.  Subsequently, the commonly used networks in AI
for change detection are described.  From a practical point of view, the
application domains of AI-based change detection methods are classified based on
their applicability. Finally, the major challenges and prospects of AI for
change detection are discussed and delineated, including (a) heterogeneous big
data processing, (b) unsupervised AI, and (c) the reliability of AI. This review
will be beneficial for researchers inunderstanding this field.

[121]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiNpL6M5PTqAhUJ2KQKHX_yA5EQFjAHegQIBxAB&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F12%2F10%2F1688%2Fpdf&usg=AOvVaw3cQq-rIhdtJpFUR-2bCIIO

### [Deep learning and superpixel feature extraction based on contractive autoencoder for change detection in sar images (2019)][119]

Image segmentation based on superpixel is used inurban and land cover change
detection for fast locating region of interest. However, the segmentation
algorithms often degrade due to speckle noise in synthetic aperture radar
images. In this paper, a feature learning method using stacked contractive
autoencoder is presented to extract temporal change feature from superpixel
with noise suppression. Firstly, an affiliated temporal change image which
obtains temporal difference in pixel level are built by three different metrics.
Secondly, the simple linear iterative clustering algorithm is used to generate
superpixels which tightly adhere to the change image boundaries for the purpose
of acquiring homogeneous change samples. Thirdly, a stacked Contractive
autoencoder (sCAE) network are trained with the superpixel samples as input to
learn the change features in semantic. And then, the encoded features by this
sCAE model are binary classified to create the change result map. Finally, the
proposed method is compared with methods based on PCA and MRF. Experiment
results show our deep learning model canseparate non-linear noise efficiently
from change features and obtain better performance in change detection for SAR
images than conventional change detection algorithms.

[119]: https://sci-hub.tw/10.1109/tii.2018.2873492

### [Feature-level change detection using deep representationand feature change analysis for multispectral imagery (2016)][118]

**Unsupervised**

Due to the noise interference and redundancy in multispectral images, it is
promising to transform the available spectral channels into a suitable feature
space for relieving noise and reducing the redundancy. The booming of deep
learning provides a flexible tool to learn abstract and invariant features
directly from the data in their raw forms. In this letter, we propose an
unsupervised change detection technique for multispectral images, in which we
combine deep belief networks (DBNs) and feature change analysis to highlight
changes. First, a DBN is established to capture the key information for
discrimination and suppress the irrelevant variations. Second, we map bitemporal
change feature into a 2-D polar domain to characterize the change information.
Finally, an unsupervised clustering algorithm is adopted to distinguish the
changed and unchanged pixels, and then, the changed types can be identified by
classifying the changed pixels into several classes according to the directions
of feature changes. The experimental results demonstrate the effectiveness and
robustness of the proposed method.

[118]: https://www.researchgate.net/publication/307610359_Feature-Level_Change_Detection_Using_Deep_Representation_and_Feature_Change_Analysis_for_Multispectral_Imagery

### [Deep learning and mapping based ternary changedetection for information unbalanced images (2017)][117]

**Unsupervised**

This paper mainly introduces a novel deep learning and mapping (DLM)framework
oriented to the ternary change detection task for information unbalanced images.
Different from the traditional intensity-based methodsavailable, the DLM
framework is based on the operation of the features ex-tracted from the two
images. Due to the excellent performance of deep learn-ing in information
representation and feature learning, two networks are usedhere. First, the
stacked denoising autoencoder is used on two images, serv-ing as a feature
extractor. Then after a sample selection process, the stackedmapping network is
employed to obtain the mapping functions, establishing the relationship between
the features for each class. Finally, a comparison between the features is made
and the final ternary map is generated through the clustering of the comparison
result. This work is highlighted by two as-pects. Firstly, previous works focus
on two images with similar properties, whereas the DLM framework is based on two
images with quite different properties, which is a usually encountered case.
Secondly, the DLM frame-work is based on the analysis of feature instead of
superficial intensity, which avoids the corruptions of unbalanced information to
a large extent. Parame-ter tests on three datasets provide us with the
appropriate parameter settings and the corresponding experimental results
demonstrate its robustness and effectiveness in terms of accuracy and time
complexity.

[117]: https://sci-hub.tw/10.1016/j.patcog.2017.01.002

### [Change detection in sar images based on deep semi-nmf and svd networks (2017)][116]

**Unsupervised**

With the development of Earth observation programs, more and more multi-temporal
synthetic aperture radar (SAR) data are available from remote sensing platforms.
Therefore, it is demanding to develop unsupervised methods for SAR image change
detection. Recently, deep-learning-based methods have displayed promising
performance for remote sensing image analysis. However, these methods can only
provide excellent performance when the number of training samples is
sufficiently large. In this paper, a novel simple method for SAR image change
detection is proposed. The proposed method uses two singular value decomposition
(SVD) analyses to learn the non-linear relations between multi-temporal images.
By this means, the proposed method can generate more representative feature
expressions with fewer samples. Therefore, it provides asimple yet effective way
to be designed and trained easily. Firstly, deep semi-non-negative matrix
factorization (Deep Semi-NMF) is utilized to select pixels that have a high
probability of being changed or unchanged as samples. Next, image patches
centered at these sample pixels are generated from the input multi-temporal SAR
images. Then, we build SVD networks, which are comprisedof two SVD convolutional
layers and one histogram feature generation layer. Finally, pixels in both
multi-temporal SAR images are classified by the SVD networks, and then the final
change map can be obtained. The experimental results of three SAR datasets have
demonstrated the effectiveness and robustness of the proposed method.

[116]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwid5brO4vTqAhUE3qQKHaiwChkQFjABegQIBhAB&url=http%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F9%2F5%2F435%2Fpdf&usg=AOvVaw1NT74-U4Y-59CTdItq-CrR

### [Nonnegative matrix factorization: A comprehensive review (2012)][115]

**Unsupervised**

Nonnegative Matrix Factorization (NMF), a relatively novel paradigm for
dimensionality reduction, has been in the ascendant since its inception. It
incorporates the nonnegativity constraint and thus obtains the parts-based
representation as well as enhancing the interpretability of the issue
correspondingly. This survey paper mainly focuses on the theoretical research
into NMF overthe last 5 years, where the principles, basic models, properties,
and algorithms of NMF along with its various modifications, extensions, and
generalizations are summarized systematically. The existing NMF algorithms are
divided into four categories: Basic NMF (BNMF),Constrained NMF (CNMF),
Structured NMF (SNMF), and Generalized NMF (GNMF), upon which the design
principles, characteristics, problems, relationships, and evolution of these
algorithms are presented and analyzed comprehensively. Some related work not on
NMF that NMF should learn from or has connections with is involved too.
Moreover, some open issues remained to besolved are discussed. Several relevant
application areas of NMF are also briefly described. This survey aims to
construct an integrated, state-of-the-art framework for NMF concept, from which
the follow-up research may benefit.

[115]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjDkLDD4_TqAhWJM-wKHc2FChUQFjACegQIBhAB&url=http%3A%2F%2Foa.ee.tsinghua.edu.cn%2F~zhangyujin%2FDownload-Paper%2FE224%3DTKDE-13.pdf&usg=AOvVaw2ZePbKt-ByYkhSGdM628JK

### [Restructuring of Deep Neural Network Acoustic Models with Singular Value Decomposition (2013)][114]

**Unsupervised**

Recently proposed deep neural network (DNN) obtains significant accuracy
improvements in many large vocabulary continuous speech recognition (LVCSR)
tasks. However, DNN requires much more parameters than traditional systems,
which brings huge cost during online evaluation, and also limits the application
of DNN in a lot of scenarios. In this paper we present our new effort on DNN
aiming at reducing the model size while keeping the accuracy improvements. We
apply singular value decomposition (SVD) on the weight matrices in DNN, and then
restructure the model based on the inherent sparseness of the original matrices.
After restructuring we can reduce the DNN model size significantly with
negligible accuracy loss. We also fine-tune the restructured model using the
regular back-propagation method to get the accuracy back when reducing the DNN
model size heavily. The proposed method has been evaluated on two LVCSR tasks,
with context-dependent DNN hidden Markov model (CD-DNN-HMM). Experimental
result show that the proposed approach dramatically reduces the DNN model size
by more than 80% without losing any accuracy

[114]: https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/svd_v2.pdf

### [Feature learning and change feature classification based on deep learning for ternary change detection in SAR images (2017)][113]

**Unsupervised**

Ternary change detection aims to detect changes and group the changes into
positive change and negative change. It is of great significance in the joint
interpretation of spatial-temporal synthetic aperture radar images. In this
study, sparse autoencoder, convolutional neural networks (CNN) and unsupervised
clustering are combined to solve ternary change detection problem without any
supervison. Firstly, sparse autoencoder is used to transform log-ratio
difference image into a suitable feature space for extracting key changes and
ssuppressing outliers and noise. And then the learned features are clustered into
three classes, which are taken as the pseudo labels for training a CNN model as
change feature classifier. The reliable training samples for CNN are selected
from the feature maps learned by sparse autoencoder with certain selection
rules. Having training samples and the corresponding pseudo labels, the CNN
model can be trained by using back propagation with stochastic gradient descent.
During its training procedure, CNN is driven to learn the concept of change, and
more powerful model is established to distinguish different types of changes.
Unlike the traditional methods, the proposed framework integrates the merits of
sparse autoencoder and CNN to learn more robust difference representations and
the concept of change for ternary change detection. Experimental results on real
datasets validate the effectiveness and superiority of the proposed framework.

[113]: https://sci-hub.tw/10.1016/j.isprsjprs.2017.05.001

### [A Generative Discriminatory Classified Network for Change Detection in Multispectral Imagery][112]

**Unsupervised**

Multispectral image change detection based on deeplearning generally needs a
large amount of training data. How-ever, it is difficult and expensive to mark a
large amount of labeled data. To deal with this problem, we propose a generative
discrimi-natory classified network (GDCN) for multispectral image change
detection, in which labeled data, unlabeled data, and new fakedata generated by
generative adversarial networks are used. The GDCN consists of a discriminatory
classified network (DCN) and a generator. The DCN divides the input data into
changed class, unchanged class, and extra class, i.e., fake class. The generator
re-covers the real data from input noises to provide additional training samples
so as to boost the performance of the DCN. Finally, the bitemporal multispectral
images are input to the DCN to get the final change map. Experimental results on
the real multispectral imagery datasets demonstrate that the proposed GDCN
trained by unlabeled data and a small amount of labeled data can achieve
competitive performance compared with existing methods

[112]: https://sci-hub.tw/10.1109/jstars.2018.2887108

### [Saliency-Guided Deep Neural Networksfor SAR Image Change Detection][111]

**Unsupervised**

Change detection is an important task to identify land-cover changes between the
acquisitions at different times. For synthetic aperture radar (SAR) images,
inherent speckle noise of the images can lead to false changed points, which
affects the change detection performance. Besides, the supervised classifier in
change detection framework requires numerous training samples, which are
generally obtained by manuallabeling. In this paper, a novel unsupervised method
named saliency-guided deep neural networks (SGDNNs) is proposed for SAR image
change detection. In the proposed method, to weaken the influence of speckle
noise, a salient region that probably belongs to the changed object is extracted
from the difference image. To obtain pseudotraining samples automatically,
hierarchical fuzzy C-means (HFCM) clustering is developed to select samples with
higher probabilities to be changed andunchanged. Moreover, to enhance the
discrimination of sample features, DNNs based on the nonnegative- and
Fisher-constrained autoencoder are applied for final detection. Experimental
results on five real SAR data sets demonstrate the effectiveness of the proposed
approach.

[111]: https://sci-hub.tw/10.1109/tgrs.2019.2913095

### [Hierarchical Unsupervised Fuzzy Clustering (2000)][110]

**Unsupervised**

A new recursive algorithm for hierarchical fuzzy partitioning is presented. The
algorithm has the advantages of hierarchical clustering, while maintaining fuzzy
clustering rules. Each pattern can have a nonzero membership in more than one
subset of the data in the hierarchy. Optimal feature extraction and reduction is
optionally reapplied for each subset. Combining hierarchical and fuzzy concepts
is suggested as a natural feasible solution to the cluster validity problem of
realdata. The convergence and membership conservation of the algorithm are
proven. The algorithm is shown to be effective for avariety of data sets with a
wide dynamic range of both covariance matrices and number of members in each
class.

[110]: https://sci-hub.tw/10.1109/91.811242

### [Unsupervised Deep Noise Modeling forHyperspectral Image Change Detection (2019)][109]

**Unsupervised**

Hyperspectral image (HSI) change detection plays an important role in remote
sensing applications, and considerable research has been done focused on
improving change detection performance. However, the high dimension of
hyperspectral data makes it hard to extract discriminative features for
hyperspectral processing tasks. Though deep convolutional neural networks (CNN)
have superior capability in high-level semantic feature learning, it is
difficult to employ CNN for change detection tasks. As a ground truth map is
usually used for the evaluationof change detection algorithms, it cannot be
directly used for supervised learning. In order to better  extract
discriminative CNN features, a novel noise modeling-based unsupervised fully
convolutional network (FCN) framework is presented for HSI change detection in
this paper. Specifically, the proposed method utilizes the change detection maps
of existing  unsupervised change detection methods to train the deep CNN, and
then removes the noise during the end-to-end training process. The main
contributions of this paper are threefold: (1) A new end-to-end FCN-based deep
network architecture for HSI change detection is presented with powerful
learning features; (2) An unsupervised noise modeling method is introduced for
the robust training of the proposed deep network; (3) Experimental results on
three datasets confirm the effectiveness of the proposed method.

[109]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiyrPLNgvXqAhXcIMUKHZ7sBbMQFjABegQIBhAB&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F11%2F3%2F258%2Fpdf&usg=AOvVaw2yWc5K3Y-K4lvtQcwVvXcD

### [Automatic building change image quality assessment in high resolutionremote sensing based on deep learning][108]

**Unsupervised**

The multi-temporal high-resolution remote sensing (HRRS) images are usually
acquired at different imaging angles, with serious noise interferences and
obvious building shadows, so that detecting thechanges of urban buildings is a
problem. In order to address this challenge, a deep learning-based algorithm
called ABCDHIDL is proposed to automatically detect the building changes from
multi-temporal HRRS images. Firstly, an automatic selection method of labeled
samples of building changes based onmorphology (ASLSBCM) is proposed. Secondly,
a deep learning model (DBN-ELM) for building changes detection based on deep
belief network (DBN) and extreme learning machine (ELM) is proposed. A
convolution operation is employed to extract the spectral, texture and spatial
features and generate a com-bined low-level features vector for each pixel in
the multi-temporal HRRS images. The unlabeled samplesare introduced to pre-train
the DBN, and the parameters of DBN-ELM are globally optimized by jointly using
the ELM classifier and the labeled samples are offered by ASLSBCM to further
improve the detection accuracy. In order to evaluate the performance of
ABCDHIDL, four groups of double-temporal WorldView2 HRRS images in four
different experimental regions are selected respectively as the test datasets,
and five other representative methods are used and compared with ABCDHIDL in the
experiments of buildings change detection. The results show that ABCDHIDL has
higher accuracy and automa-tion level than the other five methods despite its
relatively higher time consumption.

[108]: https://sci-hub.tw/10.1016/j.jvcir.2019.102585s

### [A brief introduction to weakly supervised learning (2017)][107]

**Weakly-supervised**

Supervised learning techniques construct predictive models by learning from a
large number of training examples, where each training example has a label
indicating its ground-truth output. Though current techniques have achieved
great success, it is noteworthy that in many tasks it is difficult to get strong
supervision information like fully ground-truth labels due to the high cost of
the data-labeling process. Thus,it is desirable for machine-learning techniques
to work with weak supervision. This article reviews someresearch progress of
weakly supervised learning, focusing on three typical types of weak supervision:
incomplete supervision, where only a subset of training data is given with
labels; inexact supervision, where the training data are given with only
coarse-grained labels; and inaccurate supervision, where the given labels are
not always ground-truth.

[107]: https://watermark.silverchair.com/nwx106.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAp4wggKaBgkqhkiG9w0BBwagggKLMIIChwIBADCCAoAGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMxUpF6HHLo9cpMCpkAgEQgIICUeqX5iIfxQxy8bJFcWAKBtf8ttlb-oKLc_OvfRE-pkQeAfQszKOFiGv3blyW3Ej-9ultPYj3XAPkEHL1nXwxOrWJ-AmJpVczKuAc7Zw1vqkWyPZz30nVZhA6iut2IuqCSAYqlBpyQKmunheBRz6QHtmiin_9P5qo7yT0unHMXiQOeakK9a_aTIeLwVaQjtIFxvrSYdUCB6cDJ2mfrMFpjqLQlKLoLFXZqlPdfEzrYZTmioPhN2siXcwnlU17cORyC6tlHIP9LopQ2zJt3IsmMmRCsdt5iOd47O3bNIshzO0dfxpO2VfebLWSa9FxJQIENdQYrVXAl43D5Qf_NTF71AI61qjFXh3EDe_AOAJeF-9-pFtpCZM5VRqJ2mpRmePBfkN7iUrkEpXbbM2rN34zamx_XMft9-uOwC2K1bHOHJl9yKDKPujUpgz82mIidwfsHESjv0G9lq_Whfv6DHiU-Di94wY6SKNnElzJe4zJ7mqumABbBmMQeuRUcjRsDMCDb8ravc0TuOXgUoK9HPu4ev7HNlc3xTzut2aNkKe-xec-PWpQYpmDlRVO9cI3FFklg9_zw-jn5mmEYQTz3aEmUWSRkcPY18ZzPuIZOrcZnbiBsjTeermUWCKMNwjSUWJXBjmzheNu9KxSpvtHcpb9PwCgY-f9x8X__DWJ3NOMRjKxSFTV3QqDNRbQDADhPhahsL-Ku8lefKn2VFXkmlne03t7zKqi--K3hgD3BzqgKqUXRDY6Ufgmj34_iwc6_VVZQ_KmNhJeC39ft11MF3oZ0-3U

### [Towards Safe Weakly Supervised Learning][106]

**Weakly-supervised**

In this paper, we study weakly supervised learning where a large amount of data
supervision is not accessible. This includes i) incomplete supervision, where
only a small subset of labels is given, such as semi-supervised learning and
domain adaptation; ii) inexact supervision, where only coarse-grained labels are
given, such as multi-instance learning and iii) inaccurate supervision, where
the given labels are not always ground-truth, such as label noise learning.
Unlike supervised learning which typically achieves performance improvement with
more labeled examples, weakly supervised learning may sometimes even degenerate
performance with more weakly supervised data. Such deficiency seriously hinders
the deployment of weakly supervised learning to real tasks. It is thus highly
desired to study safe weakly supervised learning, which never seriously hurts
performance. To this end, we present a generic ensemble learning scheme to
derive a safe prediction by integrating multiple weakly supervised learners. We
optimize the worst-case performance gain and lead to a maximin optimization.
This brings multiple advantages to safe weakly supervised learning. First, for
many commonly used convex loss functions in classification and regression, it is
guaranteed to derive a safe prediction under a mild condition. Second, prior
knowledge related to the weight of the base weakly supervised learners can be
flexibly embedded. Third, it can be globally and efficiently addressed by simple
convex quadratic or linear program. Finally, it is in an intuitive geometric
interpretation with the least square loss. Extensive experiments on various
weakly supervised learning tasks, including semi-supervised learning, domain
adaptation, multi-instance learning and label noise learning demonstrate our
effectiveness

[106]: https://pdfs.semanticscholar.org/9a7e/7864c2a9bcbb521c617c6f6678a7bfbb5a28.pdf

### [Weakly supervised target detection in remote sensing images based on transferred deep features and negative bootstrapping][103]

**Weakly-supervised**

Target detection in remote sensing images (RSIs) is a fundamental yet
challenging problem faced for remote sensing images analysis. More recently,
weakly supervised learning, in which training sets require only binary labels
indicating whether an image contains the object or not, has attracted
considerable attention owing to its obvious advantages such as alleviating the
tedious and time consuming work of human annotation. Inspired by its impressive
success in computer vision field, in this paper, we propose a novel and
effective framework for weakly supervised target detection in RSIs based on
transferred deep features and negative bootstrapping. On one hand, to
effectively mine information from RSIs and improve the performance of target
detection, we develop a transferred deep model to extract high-level features
from RSIs, which can be achieved by pre-training a convolutional neural network
model on a large-scale annotated dataset (e.g. ImageNet) and then transferring
i tto our task by domain-specifically fine-tuning it on RSI datasets. On the
other hand, we integrate negative bootstrapping scheme into detector training
process to make the detector converge more stably and faster by exploiting the
most discriminative training samples. Comprehensive evaluations on three RSI
datasets and comparisons with state-of-the-art weakly supervised target
detection approaches demonstrate the effectiveness and superiority of the
proposed method.

[103]: https://sci-hub.tw/10.1007/s11045-015-0370-3

### [Negative Bootstrapping for Weakly Supervised Target Detection  in Remote Sensing Images][102]

**Weakly-supervised**

When training a classifier in a traditional weakly supervised learning scheme,
negative samples are obtained by randomly sampling. However, it may bring
deterioration or fluctuation for the performance of the classifier during the
iterative training process. Considering a classifier is inclined to misclassify
negative examples which resemble positive ones, comprising these misclassified
and informative negatives should be important for enhancing the effectiveness
and robustness of the classifier. In this paper, we propose to integrate
Negative Bootstrapping scheme into weakly supervised learning framework to
achieve effective target detection in remote sensing images. Compared with
traditional weakly supervised target detection schemes, this method mainly has
three advantages. Firstly, our model training framework converges more stable
and faster by selecting the most discriminative training samples. Secondly, on
each iteration, we utilize the negative samples which are most easily
misclassified to refine target detector, obtaining better performance. Thirdly,
we employ a pre-trained convolutional neural network (CNN) model named Caffe to
extract high-level features from RSIs, which carry more semantic meanings and
hence yield effective image representation. Comprehensive evaluations on a high
resolution airplane dataset and comparisons with state-of-the-art weakly
supervised target detection approaches demonstrate the effectiveness and
robustness of the proposed method.

[102]: https://sci-hub.tw/10.1109/bigmm.2015.13

### [PGA-SiamNet: Pyramid Feature-Based Attention-Guided Siamese Network for Remote Sensing Orthoimagery Building Change Detection][120]

**Weakly-supervised**

In recent years, building change detection has made remarkable progress through
using deep learning. The core problems of this technique are the need for
additional data (e.g., Lidar or semantic labels) and the difficulty in
extracting sufficient features. In this paper, we propose an end-to-end network,
called the pyramid feature-based attention-guided Siamese network (PGA-SiamNet),
to solve these problems. The network is trained to capture possible changes
using a convolutional neural network in a pyramid. It emphasizes the importance
of correlation among the input feature pairs by introducing a global
co-attention mechanism. Furthermore, we effectively improved the long-range
dependencies of the features by utilizing various attention mechanisms and then
aggregating the features of the low-level and co-attention level; this helps to
obtain richer object information. Finally, we evaluated our method with a
publicly available dataset (WHU) building dataset and a new dataset (EV-CD)
building dataset. The experiments demonstrate that the proposed method is
effective for building change detection and outperforms the existing
state-of-the-art methods on high-resolution remote sensing orthoimages in
various metrics.

[120]: https://www.researchgate.net/publication/339048512_PGA-SiamNet_Pyramid_Feature-Based_Attention-Guided_Siamese_Network_for_Remote_Sensing_Orthoimagery_Building_Change_Detection

### [The Spectral-Spatial Joint Learning for Change Detection in Multispectral Imagery (2019)][101]

Change detection is one of the most important applications in the remote-sensing
domain. More and more attention is focused on deep neural network based change
detection methods. However, many deep neural networks based methods did not take
both the spectraland spatial information into account. Moreover, the underlying
information of fused features is not fully explored. To address the
above-mentioned problems, a Spectral-Spatial Joint Learning Network (SSJLN) is
proposed. SSJLN contains three parts: spectral-spatial joint representation,
feature fusion, and discrimination learning. First, the spectral-spatial joint
representation is extracted from the network similar to the Siamese CNN (S-CNN).
Second, the above-extracted features are fused to represent the difference
information that proves to be effective for the change detection task. Third,
the discrimination learning is presented to explore the underlying information
of obtained fused features to better represent the discrimination. Moreover, we
present a new loss function that considers both the losses of the
spectral-spatial joint representation procedure and the discrimination learning
procedure. The effectiveness of our proposed SSJLN is verified on four real
datasets. Extensive experimental results show that our proposed SSJLN can
outperform the other state-of-the-art change detection methods.

[101]: https://pdfs.semanticscholar.org/fa6f/19a1c4a8cb1f30e5af13a010941ad03f0098.pdf

### [The Spectral-Spatial Joint Learning for ChangeDetection in Multispectral Imagery (2019)][100]

Change detection is one of the most important applications in the remote
sensing domain. More and more attention is focused on deep neural network based
change detection methods. However, many deep neural networks based methods did
not take both the spectral and spatial information into account. Moreover, the
underlying information of fused features is not fully explored. To address the
above-mentioned problems, a Spectral-Spatial Joint Learning Network (SSJLN) is
proposed. SSJLN contains three parts:  spectral-spatial joint representation,
feature fusion, and discrimination learning. First, the spectral-spatial joint
representation is extracted from the network similar to the Siamese CNN (S-CNN).
Second, the above-extracted features are fused to represent the difference
information that proves to be effective for the change detection task. Third,
the discrimination learning is presented to explore the underlying information
of obtained fused features to better represent the discrimination. Moreover, we
present a new loss function that considers both the losses of the
spectral-spatial joint representation procedure and the discrimination learning
procedure. The effectiveness of our proposed SSJLN is verified on four realdata
sets. Extensive experimental results show that our proposed SSJLN can outperform
the other state-of-the-art change detection methods.

[100]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjc79_jqvLqAhWDGewKHd9ZB5gQFjABegQIARAB&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F11%2F3%2F240%2Fpdf&usg=AOvVaw3IaItVELvrq8jVpGl3uQ8K

### [Multitask Learning for Large-scale Semantic Change Detection (2018)][99]

Change detection is one of the main problems in remote sensing, and is essential
to the accurate processing and understanding of the large scale Earth
observation data available through programs such as Sentinel and Landsat. Most
of the recently proposed change detection methods bring deep learning to this
context, but openly available change detection datasets are still very scarce,
which limits the methods that can be proposed and tested. In this paper we
present the first large scale high resolution semantic change detection (HRSCD)
dataset, which enables the usage of deep learning methods for semantic change
detection. The dataset contains coregistered RGB image pairs, pixel-wise change
information and land cover information. We then propose several methods using
fully convolutional neural networks to perform semantic change detection. Most
notably, we present a network architecture that performs change detection and
land cover mapping simultaneously, while using the predicted land cover
information to help to predict changes. We also describe a sequential training
scheme that allows this network to be trained without setting a hyperparameter
that balances different loss functions and achieves the best overall results.

[99]: https://arxiv.org/abs/1810.08452

### [Detecting Large-Scale Urban Land Cover Changes from Very High Resolution Remote Sensing Images Using CNN-Based Classification (2019)][98]

 The study investigates land use/cover classification and change detection of
 urban areas from very high resolution (VHR) remote sensing images using deep
 learning-based methods. Firstly, we introduce a fully Atrous convolutional
 neural network (FACNN) to learn the land cover classification. In the FACNN
 an encoder, consisting of full Atrous convolution layers, is proposed for
 extracting scale robust features from VHR images. Then, a pixel-based change
 map is produced based on the classification map of current images and an
 outdated land cover geographical information system (GIS) map. Both
 polygon-based and object-based change detection accuracy is investigated,
 where a polygon is the unit of the GIS map and an object consists of those
 adjacent changed pixels on the pixel-based change map. The test data covers a
 rapidly developing city of Wuhan (8000 km2), China, consisting of 0.5 m ground
 resolution aerial images acquired in 2014, and 1 m ground resolution Beijing-2
 satellite images in 2017, and their land cover GIS maps. Testing results showed
 that our FACNN greatly exceeded several recent convolutional neural networks in
 land cover classification. Second, the object-based change detection could
 achieve much better results than a pixel-based method, and provide accurate
 change maps to facilitate manual urban land cover updating.

[98]: https://www.mdpi.com/2220-9964/8/4/189

### [Dual-Dense Convolution Network for Change Detection of High-Resolution Panchromatic Imagery (2018)][97]

This paper presents a robust change detection algorithm for high-resolution
panchromatic imagery using a proposed dual-dense convolutional network (DCN). In
this work, a joint structure of two deep convolutional networks with dense
connectivity in convolution layers is designed in order to accomplish change
detection for satellite images acquired at different times. The proposed network
model detects pixel-wise temporal change based on local characteristics by
incorporating information from neighboring pixels. Dense connection in
convolution layers is designed to reuse preceding feature maps by connecting
them to all subsequent layers. Dual networks are incorporated by measuring the
dissimilarity of two temporal images. In the proposed algorithm for change
detection, a contrastive loss function is used in a learning stage by running
over multiple pairs of samples. According to our evaluation, we found that the
proposed framework achieves better detection performance than conventional
algorithms, in area under the curve (AUC) of 0.97, percentage correct
classification (PCC) of 99%, and Kappa of 69, on average.

[97]: https://www.mdpi.com/2076-3417/8/10/1785

### [GETNET: A General End-to-end Two-dimensional CNN Framework for Hyperspectral Image Change Detection (2019)][96]

Change detection (CD) is an important application of remote sensing, which provides timely change information about large-scale Earth surface. With the emergence of hyperspectral imagery, CD technology has been greatly promoted, as hyperspectral data with the highspectral resolution are capable of detecting finer changes than using the traditional multispectral imagery. Nevertheless, the high dimension of hyperspectral data makes it difficult to implement traditional CD algorithms. Besides, endmember abundance information at subpixel level is often not fully utilized. In order to better handle high dimension problem and explore abundance information, this paper presents a General End-to-end Two-dimensional CNN (GETNET) framework for hyperspectral image change detection (HSI-CD). The main contributions of this work are threefold: 1) Mixed-affinity matrix that integrates subpixel representation is introduced to mine more cross-channel gradient features and fuse multi-source information; 2) 2-D CNN is designed to learn the discriminative features effectively from multi-source data at a higher level and enhance the generalization ability of the proposed CD algorithm; 3) A new HSI-CD data set is designed for the objective comparison of different methods. Experimental results on real hyperspectral data sets demonstrate the proposed method outperforms most of the state-of-the-arts.


[96]: https://arxiv.org/abs/1905.01662

### [End-to-End Change Detection for High Resolution Satellite Images Using Improved UNet++ (2019)][95]

Change detection (CD) is essential to the accurate understanding of land surface
changes using available Earth observation data. Due to the great advantages in
deep feature representation and nonlinear problem modeling, deep learning is
becoming increasingly popular to solve CD tasks in remote-sensing community.
However, most existing deep learning-based CD methods are implemented by either
generating difference images using deep features or learning change relations
between pixel patches, which leads to error accumulation problems since many
intermediate processing steps are needed to obtain final change maps. To address
the above-mentioned issues, a novel end-to-end CD method is proposed based on an
effective encoder-decoder architecture for semantic segmentation named UNet++,
where change maps could be learned from scratch using available annotated
datasets. Firstly, co-registered image pairs are concatenated as an input for
the improved UNet++ network, where both global and fine-grained information can
be utilized to generate feature maps with high spatial accuracy. Then, the
fusion strategy of multiple side outputs is adopted to combine change maps from
different semantic levels, thereby generating a final change map with high
accuracy. The effectiveness and reliability of our proposed CD method are
verified on very-high-resolution (VHR) satellite image datasets. Extensive
experimental results have shown that our proposed approach outperforms the
other state-of-the-art CD methods.



### [Deep Learning for Change Detection in Remote Sensing Images: Comprehensive Review and Meta-Analysis][94]



Deep learning (DL) algorithms are considered as a methodology of choice for
remote-sensing image analysis over the past few years. Due to its effective
applications, deep learning has also been introduced for automatic change
detection and achieved great success. The present study attempts to provide a
comprehensive review and a meta-analysis of the recent progress in this subfield.
Specifically, we first introduce the fundamentals of deep learning methods which
are frequently adopted for change detection. Secondly, we present the details of
the meta-analysis conducted to examine the status of change detection DL studies.
Then, we focus on deep learning-based change detection methodologies for remote
sensing images by giving a general overview of the existing methods.
Specifically, these deep learning-based methods were classified into three
groups; fully supervised learning-based methods, fully unsupervised
learning-based methods and transfer learning-based techniques. As a result of
these investigations, promising new directions were identified for future
research. This study will contribute in several ways to our understanding of
deep learning for change detection and will provide a basis for further research.



### [Code-Aligned Autoencoders for Unsupervised Change Detection in Multimodal Remote Sensing Images (2020)][92]


Abstract—Image translation with convolutional autoencodershas recently been used
as an approach to multimodal change detection in bitemporal satellite images. A
main challenge is the alignment of the code spaces by reducing the contribution
of change pixels to the learning of the translation function. Many existing
approaches train the networks by exploiting supervised information of the change
areas, which, however, is not always available. We propose to extract relational
pixel information captured by domain-specific affinity matrices at the input and
use this to enforce alignment of the code spaces and reduce the impact of change
pixels on the learning objective. A change prior is derived in an unsupervised
fashion from pixel pair affinities thatare comparable across domains. To achieve
code space alignment we enforce that pixel with similar affinity relations in
the input domains should be correlated also in code space. We demonstrate the
utility of this procedure in combination with cycle consistency. The proposed
approach are compared with state-of-the-art deep learning algorithms.
Experiments conducted on four real datasets show the effectiveness of our
methodology.

** promising **

[code][https://github.com/llu025/Heterogeneous_CD]

### [Change Detection in Heterogeneous Optical and SAR Remote Sensing Images via Deep Homogeneous Feature Fusion (2020)][93]

Change detection in heterogeneous remote sensing images is crucial for disaster
damage assessment. Recent methods usehomogenous transformation, which transforms
the heterogeneous optical and SAR remote sensing images into  the same feature
space, to achieve change detection. Such transformations mainly operate on the
low-level feature space and may corrupt the semantic content, deteriorating the
performance of change detection. To solve this problem, this paper presents a
new homogeneous transformation model termed deep homogeneous feature fusion
(DHFF) based on image style transfer (IST). Unlike the existing methods, the
DHFF method segregates the semantic content and the style features in the
heterogeneous images to perform homogeneous transformation. The separation of
the semantic content and the style in homogeneous transformation prevents the
corruption of image semantic content, especially  in  the  regions of change.
In this way, the detection performance is improved with accurate homogeneous
transformation. Furthermore, we present a new iterative IST (IIST) strategy,
where the cost function in each IST iteration measures and thus maximizes the
feature homogeneity in additional new feature subspaces for change detection.
After that, change  detection is accomplished accurately on the original and
the transformed images that are in the same feature space. Real remote
sensing imagesa cquired by SAR and optical satellites are utilized to evaluate
the performance of the proposed method. The experiments demonstrate that the
proposed DHFF method achieves significant improvement for change detection in
heterogeneous optical and SAR remote sensing images, in terms of both accuracy
rate and Kappa index.

[93]:https://arxiv.org/pdf/2004.03830.pdf


### [From W-Net to CDGAN: Bi-temporal Change Detection via Deep Learning Techniques (2020)][95]

Traditional change detection methods usually follow the image differencing,
change feature extraction and classification framework, and their performance
is limited by such simple image domain differencing and also the hand-crafted
features. Recently, the success of deep convolutional neural networks (CNNs)
has widely spread across the whole field of computer vision for their powerful
representation abilities. In this paper, we therefore address the remote sensing
image change detection problem with deep learning techniques. We firstly propose
an end-to-end dual-branch architecture, termed as the W-Net, with each branch
taking as input one of the two  bi-temporal images as in the traditional change
detection models. In this way, CNN features with more powerful representative
abilities can be obtained to boost the final detection performance. Also, W-Net
performs differencing in the feature domain rather than in the traditional image
domain, which greatly alleviates loss of useful information for determining the
changes. Furthermore, by reformulating change detection as an image translation
problem, we apply the recently popular Generative Adversarial Network (GAN) in
which our W-Net serves as the Generator, leading to a new GAN architecture for
change detection which we call CDGAN. To train our networks and also facilitate
future research, we construct a large scale dataset by collecting images from
Google Earth and provide carefully manually annotated ground truths. Experiments
show that our proposed methods can provide fine-grained change detection results
superior to the existing state-of-the-art baselines.

[95]: https://arxiv.org/pdf/2003.06583.pdf

### [GRAPH-BASED FUSION FOR CHANGE DETECTION IN MULTI-SPECTRAL IMAGES][94]

In this paper we address the problem of change detectionin  multi-spectral  images  by  proposing  a  data-driven  frame-work of graph-based data fusion.  The main steps of the pro-posed approach are:  (i) The generation of a multi-temporalpixel based graph, by the fusion of intra-graphs of each tem-poral  data;  (ii)  the  use  of  Nystr ̈om  extension  to  obtain  theeigenvalues and eigenvectors of the fused graph, and the se-lection of the final change map. We validated our approach intwo real cases of remote sensing according to both qualitativeand quantitative analyses. The results confirm the potential ofthe proposed graph-based change detection algorithm outper-forming state-of-the-art methods.

[94]: https://arxiv.org/pdf/2004.00786.pdf

### [Urban Change Detection for Multispectral Earth Observation Using CNNs][31]

Presents the [Onera Satellite Change Detection Dataset][32] (Sentinel-2 pre/post
change image pairs).

Studies performance of two network architectures on dataset, determining the
presence of urban changes pixel-wise.

#### Inputs:

Uses pairs of before/after 15x15xC image patches as input, predicting central
pixel binary probability for (urban) change/no-change. Idea is for network to
learn from context how to ignore changes due to natural processes.

For tractability full change maps generated by larger strides than each pixel,
then a 2D-Gaussian assumption about change of surrounding pixels.

Different resolution channels (10-60m) are handled by upsampling lower res.

#### Model structure:

Two networks are compared:

- **Early Fusion (EF)**: Concatenate the two image pairs as the first
step of the network.  The input of the network can then be seen as a single
patch of 15x15x2C, which is then processed by a series of seven convolutional
layers and two fully connected layers, where the last layer is a softmax layer
with two outputs associated with the classes of change and no change.

- **Siamese (Siam) network**: Process each of the patches in parallel by two
branches of four convolutional layers with shared weights,concatenating the
outputs and using two fully connected layers to obtain two output values as
before.

#### Results:
- **Early Fusion network generally better** than Siamese network
- **70-90% Accuracy**  (per class)
- **3->13 channels => <~5% accuracy improvement**

#### Additional observations:
- Transfer learning approaches limited by fact that pretrained models are trained
using RGB input, while sentinel-3 images include 13 bands.

- OSM unreliable for change detection since addition date != building date, older
maps unavailable.

### [High-resolution optical remote sensing imagery change detection through deep transfer learning (2018)][65]

Proposes **unsupervised** change-detection approach for optical images based on
CNNs which learn transferable features which are invariant to
contrast/illumination changes between tasks.

Uses pre-trained AlexNet, PlacesNet, Network in Network and VGG-16 models.

Describes two frameworks, a fast one and an accurate one.

#### Inputs:

Pairs of co-registered images separated in time.

#### Model structure:

In the preprocessing phase, a geometric registration and radiometric correction
are done.

A naive pixel-wise change map is used in an intermediate step. This is defined
by forming a 6-band images stacking the channel-wise difference and log-ratio of
the pair of temporally displaced images (which are insensitive to sun angle etc).

This image is PCA'd, selecting the most important components (linear
combinations of pixels) which are classified into two classes by using the
K-means algorithm to get a naive binary change map, CM0.

##### Fast Framework

Given the pair of co-registered, temporally displaced images:

- Construct two sets of hierarchical hyperfeatures, by passing the pair
through a pre-trained FCN.

- Each set is extracted, upsampled if necessary to the input image size,
and concatenated forming a very high dimensional set of hyperfeatures

- Dimensionality reduction must be applied to remove unrepresentative features
and make computation more tractable. This is accomplished by convolving the
hyperfeatures with CMO to get a vector of (hyperfeature_depth x n_layers)
features, which can be pruned down to K x n_layers, where K is a constant value
for all layers.

- K-means with two classes is used to separate changed from unchanged regions,
using the Euclidean distance between hyperfeatures as a metric.

- *fast* but *imprecise boundary delineation* due to unpooling operation used in  
upsampling

##### Accurate Framework

Given the same inputs:

*similar to above, but with 2D gaussian windowing to get multiple ROIs*

#### Results:

Both frameworks **outperform previous state-of-the-art methods** at identifying
changed regions when presented with an identified ground truth.

**NOTE: The previous "state-of-the-art" methods referred to are <= 2015 and
don't use CNNs.**

Pre-trained VGG-16 performs best, quoting kappa = 0.7 and 0.9 respectively for
fast and accurate frameworks respectively.

Overall error tends to be more evenly spread between FP and FN than previous
algorithms.

#### Additional observations:

Choice of data representation is key to success of CD algorithms.

### [A Deep Convolutional Coupling Network for Change Detection Based on Heterogeneous Optical and Radar Images (2016)][69]

Proposes an unsupervised deep convolutional coupling network for change
detection based on pairs of heterogeneous images acquired by optical sensors
and SAR images on different dates.

Heterogeneous images are captured by different types of sensors with different
characteristics, e.g., optical and radar sensors.
Homogeneous images are acquired by homogeneous sensors between which the
intensity of unchanged pixels are assumed to be linearly correlated.

#### Inputs:

Pairs of co-registered, *heterogeneous*, denoised images of the same scene
temporally displaced.

#### Model structure:

*Will fill this in if using heterogeneous images becomes important.*

#### Results:

*Will fill this in if using heterogeneous images becomes important.*

#### Additional observations:

Different sensors may capture distinct statistical properties of the same ground
 object, and thus, inconsistent representations of heterogeneous images may make
 change detection more difficult than using homogeneous images.

### [Unsupervised Change Detection in Satellite Images Using CNNs (Late 2018)][19]

**Promising!**

Proposes **semi-unsupervised** method for detecting changes between pairs of
temporally displaced images.

Comprised of CNN trained for *semantic segmentation* to extract compressed image
features, and generates an effective difference image from the feature map
information without explicitly training on difference images. Uses **U-Net**.

Uses [ISPRS Vaihingen dataset][57].

Aims to classify nature of change automatically with semantic segmentation,
while being noise resistant.

#### Inputs:

1. *Training phase*: Images with classification masks. Omitted if pre-trained
U-net available. Only buildings, immutable surfaces and background classes used.
2. *Inference phase*: Pairs of images of the same locations, temporally
displaced. 320x320 px x 3 channels.

#### Model structure:

- Feature maps are generated by the U-Net encoder for each image.

- A difference image is created for each feature map using a fixed algorithm
with a fixed threshold cutoff which determines whether the DI value is zero
or the value of the activation in the second (most recent) image.

- The five DIs are used by the decoder in the copy-and-concatenate operation

- The model outputs a semantically segmented visual rendering of the DIs.

- Threshold values for the difference-zeroing were determined by empirical
testing.

#### Results:

The success of the proposed change detection method  heavily relies on
the ability of the trained model to perform semantic segmentation.

*Test images were constructed manually by cutting and pasting over images,
and don't reflect change from natural illumination and weather processes.*

- **Semantic Segmentation**: Average classification **accuracy was 89.2%**

- **Change Detection**: The model was able to detect the location of the change
and classify it to the correct semantic class for the **91.2%** of test pairs.
*Accuracy declined as total number of pixels changed increased, owing to the
consequently different feature map activations => threshold value in
differencing includes unwanted changes*. Robust to gaussian noise.

Performance can potentially be improved by cutting the high-dimensional
satellite images into overlapping subsets, and combining change detection
signals generated by overlapping areas.

#### Additional observations:

A source of variance may arise from the translation and rotation of two images,
or from a difference in the angleat which the images were captured. This can
have a significanteffect on the accuracy of a DI unless accounted for.
[Orthorectification methods][70] can be used to compensate for sensor
orientation, and consist of geometric transformations to a mutual coordinate
system.

Testing the model on real, as opposed to simulated change, is a necessary step
to confirm that the results of this pilot study hold in more realistic
environments.

Additional experimentation should be done to test resistance to angle,
translation, and rotation differences between two images.  

### [Land Cover Change Detection via Semantic Segmentation][72]

#### Inputs:
#### Model structure:
#### Results:
#### Additional observations:

### [Change Detection between Multimodal Remote Sensing Data Using Siamese CNN][20]

#### Inputs:
#### Model structure:
#### Results:
#### Additional observations:

### [Convolutional Neural Network Features Based Change Detection in Satellite Images][68]

**This paper appears to be an older article from the same author as [65][65],
sharing many characteristics.**

### [Zoom out CNNs Features for Optical Remote Sensing Change Detection][58]

**This paper is unclear.**

Presents unsupervised change detection network based on an ImageNet-pretrained
CNN and superpixel (SLIC) segmentation technique. Uses QuickBird and Google
Earth images.

[Superpixels and SLIC][66]

#### Inputs:

Bi-temporal images.

#### Model structure:

Images are first segmented into superpixels using SLIC. PCA is applied to
extract three uncorrelated channels. Median filter applied to smooth image and
eliminate noise, followed by a bilateral filter to preserve the edges.

Each region subjected to three levels of zoom-out which are separately passed
through a pre-trained CNN, starting from the superpixel itself to regions around.

Each zoom passed through stack of convolutional layers.
Features extracted from each zoom level belonging to the same superpixel and
concatenated.

Concatenated features compared to get final change map according to some
dissimilarity measure

#### Results:

*will fill this in if I get round to decrypting the paper*

#### Additional observations:

Change detection divided into three broad classes depending on analysis unit:
pixel, kernel and object-based approaches. Latter more popular due to more
sophisticated feature extraction.

Object-based change detection split naturally into three categories:

  1. *Object overlay*: Segment one of the multi-temporal images and superimpose
     boundaries on second image. Compare each object region. Disadvantage is
     that gemoetry of objects imitates only one of the multi-temporal images.
  2. *Image-object direct comparison*: Segment each image separately, then
     compare objects from the same location. Introduces problem of 'silver'
     objects under inconsistent segmentations.
  3. *Multi-temporal image objects*: Images stacked and co-segmented in one
      step.


### [Change Detection in Synthetic Aperture Radar Images Based on DNNs (2015)][71]

According to [Unsupervised Change Detection in Satellite Images Using CNNs (Late 2018)][19]:

*"Make use of unsupervised feature learning performed by a CNN to learn the
representation of the relationship between two images. The CNN is then
fine-tuned withsupervised learning to learn the concepts of the changed and
the  unchanged  pixels. During supervised learning, a change detection map,
created by other means, is used to represent differences and similarities
between two images on a per-pixel level. Once the network is fully trained, it
is able to produce a change map directly from two given images without having
to generate a DI."

"While this approach achieves good results, it requires the creation of accurate
change maps by other means for each image pair prior to the learning process.
This makes the training of the network an expensive and time-consuming  process.
Change detection is also formulated as a binary classification problem, as it
only classifies pixels aschanged or not changed, and does not classify the
nature of the change, as the present study sets out to do."*




--------------------------------------------------------------------------------

# Time-series/sequence (spatiotemporal) prediction

*TODO*

## Datasets

*TODO*

## Papers

### [Deep-STEP: A Deep Learning Approach for Spatiotemporal Prediction of Remote Sensing Data][63]

### [A high-performance and in-season classification system of field-level crop types using time-series Landsat data and a machine learning approach][64]

### [Multi-Temporal Land Cover Classification with Sequential Recurrent Encoders][18]

--------------------------------------------------------------------------------
# Super-resolution and pansharpening

## Papers

### [Learned Spectral Super-Resolution][21]
### [Pansharpening by CNN][22]
### [Pansharpening via Detail Injection Based Convolutional Neural Networks][23]

--------------------------------------------------------------------------------

# Generative models for data synthesis

## Papers

### [GANs for Realistic Synthesis of hyperspectral samples][24]

--------------------------------------------------------------------------------
[1]: https://www.researchgate.net/publication/311223153_Speedaccuracy_trade-offs_for_modern_convolutional_object_detectors
[2]: https://arxiv.org/pdf/1603.06201.pdf
[3]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=2ahUKEwiDlo67jY3mAhWFsKQKHcFGBxMQwqsBMAF6BAgLEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DnDPWywWRIRo&usg=AOvVaw251YCv68Wl_c-eUBdnE5h-
[4]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-3-transfer-learning/
[5]: https://towardsdatascience.com/review-ssd-single-shot-detector-object-detection-851a94607d11
[6]: https://towardsdatascience.com/review-dssd-deconvolutional-single-shot-detector-object-detection-d4821a2bbeb5
[7]: https://medium.com/data-from-the-trenches/object-detection-with-deep-learning-on-aerial-imagery-2465078db8a9
[9]: https://deepsense.ai/deep-learning-for-satellite-imagery-via-image-segmentation/
[10]: https://arxiv.org/pdf/1703.06870.pdf
[11]: https://dlt18.sciencesconf.org/data/Audebert.pdf
[12]: https://arxiv.org/pdf/1511.05641.pdf
[13]: https://www.researchgate.net/publication/311223153_Speedaccuracy_trade-offs_for_modern_convolutional_object_detectors
[14]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-4-state-of-the-art/
[15]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-1-deeplab-v3/
[16]: https://www.novatec-gmbh.de/blog/semantic-segmentation-part-2-training-u-net/
[17]: https://www.researchgate.net/publication/319955230_Deep_Learning_in_Remote_Sensing_A_Review
[18]: https://arxiv.org/abs/1802.02080
[19]: https://arxiv.org/pdf/1812.05815.pdf
[20]: https://arxiv.org/pdf/1807.09562.pdf
[21]: https://arxiv.org/abs/1703.09470
[22]: https://www.researchgate.net/publication/305338139_Pansharpening_by_Convolutional_Neural_Networks
[23]: https://ieeexplore.ieee.org/document/8667040
[24]: https://arxiv.org/abs/1806.02583
[25]: https://www.mdpi.com/2072-4292/9/4/368/htm
[26]: https://arxiv.org/pdf/1806.04331.pdf
[27]: https://www.mdpi.com/2072-4292/9/1/67/htm
[28]: https://elib.dlr.de/106352/2/CNN.pdf
[29]: https://arxiv.org/pdf/1711.08681.pdf
[30]: https://arxiv.org/abs/1609.06846
[31]: http://rcdaudt.github.io/files/2018igarss-change-detection.pdf
[32]: https://rcdaudt.github.io/oscd/
[33]: https://project.inria.fr/aerialimagelabeling/
[34]: https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&ved=2ahUKEwj9nLaY0o_mAhUBIVAKHTAsAhcQFjAAegQIARAC&url=https%3A%2F%2Fwww.mdpi.com%2F2072-4292%2F11%2F15%2F1774%2Fpdf&usg=AOvVaw3BMBIIkxymYokfm6HOQSJU
[35]: https://www.cs.toronto.edu/~vmnih/data/
[36]: https://spacenetchallenge.github.io/datasets/spacenet-OffNadir-summary.html
[37]: https://medium.com/the-downlinq/introducing-the-spacenet-off-nadir-imagery-and-buildings-dataset-e4a3c1cb4ce3
[38]: https://community.topcoder.com/longcontest/stats/?module=ViewOverview&rd=17313
[39]: https://medium.com/the-downlinq/the-good-and-the-bad-in-the-spacenet-off-nadir-building-footprint-extraction-challenge-4c3a96ee9c72
[40]: https://github.com/SpaceNetChallenge/SpaceNet_Off_Nadir_Solutions
[41]: https://medium.com/the-downlinq/the-spacenet-challenge-off-nadir-buildings-introducing-the-winners-b60f2b700266
[42]: https://arxiv.org/abs/1801.05746
[43]: https://arxiv.org/abs/1709.01507
[44]: https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection/data
[45]: http://deepglobe.org/
[46]: https://github.com/tensorflow/models/tree/master/research/object_detection
[47]: http://xviewdataset.org/
[48]: https://insights.sei.cmu.edu/sei_blog/2019/01/deep-learning-and-satellite-imagery-diux-xview-challenge.html
[49]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7536139
[50]: http://dase.grss-ieee.org/
[51]: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7587405
[52]: https://core.ac.uk/download/pdf/147323889.pdf
[53]: https://www.sipeo.bgu.tum.de/downloads
[54]: https://reader.elsevier.com/reader/sd/pii/S0924271618300352?token=433A52C26A0BD7DE1E20DB752317DD7CDBE7A4CAFEFD688AFFFA429BF50362587FF40736B2A13D42FE938AFD616B9F62
[55]: https://drive.google.com/file/d/14WxHQBiFiHMH9_Xzv2BpS_w6urgA3scb/view
[56]: https://downloads.greyc.fr/vedai/
[57]: http://www2.isprs.org/commissions/comm2/wg4/vaihingen-2d-semantic-labeling-contest.html
[58]: https://www.researchgate.net/publication/318574881_Zoom_out_CNNs_features_for_optical_remote_sensing_change_detection
[59]: https://sci-hub.se/10.1109/rsip.2017.7958815
[60]: https://sci-hub.se/10.1109/tgrs.2016.2601622
[61]: https://iges.or.jp/en/publication_documents/pub/peer/en/6898/Ma+et+al+2019.pdf
[62]: https://github.com/geoslegend/Deep-Learning-for-Spatio-temporal-Prediction
[63]: https://ieeexplore.ieee.org/document/7752890
[64]: https://sci-hub.se/10.1016/j.rse.2018.02.045
[65]: https://www.researchgate.net/publication/337146658_High-Resolution_Optical_Remote_Sensing_Imagery_Change_Detection_Through_Deep_Transfer_Learning
[66]: https://medium.com/@darshita1405/superpixels-and-slic-6b2d8a6e4f08
[67]: https://sci-hub.se/10.1109/lgrs.2016.2611001
[68]: https://sci-hub.se/10.1117/12.2243798
[69]: https://sci-hub.se/10.1109/tnnls.2016.2636227
[70]: https://www.researchgate.net/publication/224999140_Comparison_of_orthorectification_methods_suitable_for_rapid_mapping_using_direct_georeferencing_and_RPC_for_optical_satellite_data
[71]: https://sci-hub.se/10.1109/tnnls.2015.2435783
[72]: https://arxiv.org/abs/1911.12903
[73]: http://www2.isprs.org/commissions/comm3/wg4/2d-sem-label-potsdam.html
[74]: https://www.kaggle.com/lopuhin/full-pipeline-demo-poly-pixels-ml-poly
[75]: http://www.grss-ieee.org/community/technical-committees/data-fusion/2015-ieee-grss-data-fusion-contest/
[76]: https://arxiv.org/abs/1904.05730
[77]: https://www.sipeo.bgu.tum.de/downloads
[78]: https://arxiv.org/pdf/1803.09050.pdf
[79]: https://www.researchgate.net/publication/327470305_PSI-CNN_A_Pyramid-Based_Scale-Invariant_CNN_Architecture_for_Face_Recognition_Robust_to_Various_Image_Resolutions
[80]: http://www.lirmm.fr/ModuleImage/Ienco.pdf
[81]: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6767260/
[82]: http://cfg.mit.edu/sites/cfg.mit.edu/files/learning_to_zoom.pdf
[83]: https://www.jeremyjordan.me/semantic-segmentation/
[84]: https://arxiv.org/pdf/1711.10684.pdf
[85]: http://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
[86]: https://arxiv.org/abs/1905.11946v1
[87]: http://proceedings.mlr.press/v102/kervadec19a/kervadec19a.pdf
[88]: https://arxiv.org/abs/1602.06564
[89]: https://arxiv.org/pdf/1707.03237.pdf
[90]: https://sci-hub.se/10.1109/tmi.2006.880587
[91]: https://arxiv.org/pdf/2004.07018.pdf
[92]: https://arxiv.org/pdf/2004.07011.pdf
[93]: https://arxiv.org/pdf/2005.01094.pdf
[94]: https://arxiv.org/abs/2006.05612
[95]: https://www.mdpi.com/2072-4292/11/11/1382/htm
