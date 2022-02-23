Models
======

The current model of choice is ``Segmentalist`` which builds on the `DeepResUNet`_ architecture 
by adding various optional additional architectural components.
The following sections will give an overview of each of these. Using this model with 
any or all these optional components will be covered in the section :ref:`segmentalist`.

For completeness, the relevant schematics from various publications will be reproduced 
here to make this page a quick reference. For full information, the included links to the source 
articles should be consulted.

DeepResUNet
-----------

The DeepResUNet architecture is described in the paper `Semantic Segmentation of Urban Buildings from VHR Remote Sensing Imagery Using a Deep Convolutional Neural Network (Yi et al. 2019)`_.

This is a fully convolutional encoder-decoder segmentation network with residual convolution blocks 
(as in ResNet) and skip connections (as in U-Net). 

.. figure:: media/drunet.png
  :alt: The DeepResUNet architecture block

  The DeepResUNet architecture.

It should be noted that from experiments, the pattern chosen in this paper for fixing the filter size 
and number of channels within each residual block leads to a significant performance increase over 
the vanilla ResNet residual block for segmentation tasks.

.. figure:: media/drunet_resblock.png
  :alt: The DeepResUNet architecture

  The residual block structure used in DeepResUNet (d) as compared with ResNet (c).

This network architecture forms the backbone of :ref:`segmentalist`, i.e. is equivalent to it with 
the following additional features disabled.

Optional Model Features
-----------------------

Spatial Attention Gates
^^^^^^^^^^^^^^^^^^^^^^^

The Spatial Attention Gate modules implemented are described in the paper 
`Attention U-Net\: Learning Where to Look for the Pancreas (Oktay et al. 2018)`_.

Spatial attention gates provide a mechanism through which the important regions 
of the encoder feature maps can be enhanced or suppressed depending on the more 
abstract semantic content of the decoder at the corresponding spatial locations.

These use use kernel-size one convolutions to project each 
set of encoder feature maps into a new space of "key" vectors at each spatial 
location, and decoder feature maps into a new space of "query" vectors at each 
spatial location. These feature maps are then used to derive an (additive) 
attention map by adding these and applying a further 1D convolution with sigmoid 
activation. This is then multiplied with the encoder feature maps, adaptively 
rescaling them according to the content of the decoder feature maps. 

.. figure:: media/sag.png
  :alt: The Spatial Attention Gate module

  The Spatial Attention Gate module. ``x`` and ``g`` are feature maps coming through 
  the encoder skip lines and from the deeper decoder stages respectively. These 
  are mapped into a key, query space with kernel-size 1 convolutions, added and 
  the a ReLU nonlinearity applied to the result. A final convolution with sigmoid 
  activation is used to generate a spatial attention map which then reweights ``x``.

The spatial attention gates are inserted immediately after each decoder upsampling 
block and intercept the encoder feature maps through the skip lines before these 
enter the next decoder block. 

.. figure:: media/sag_architecture.png
  :alt: The Spatial Attention Gate architecture

  A U-Net with spatial attention gates positioned before each intermediate decoder 
  block.

These have been demonstrated to improve encoder-decoder type models in various 
segmentation tasks.

Deep Supervision
^^^^^^^^^^^^^^^^

The version of deep supervision implemented follows the version described in the paper 
`Improving CT Image Tumor Segmentation Through Deep Supervision and Attentional Gates (Tureckova et al. 2020)`_.

Deep supervision is a mechanism to force each decoder block of the network to take on 
a more concrete role, namely learning to produce outputs which more directly correspond 
to the target segmentation map at that block's spatial resolution. It is known to improve 
results and speed up training in segmentation tasks and is used heavily in biomedical imaging.

.. figure:: media/ds.jpg
  :alt: The Deep Supervision network

  A Deep Supervision network - each decoder output feature map is projected into the same 
  space used to generate the final segmentation map and these are added (with upsampling).

The version implemented captures the output of the intermediate decoder feature maps and 
projects these into the same (channel) space as the final output feature maps used to 
generate the segmentation (before application of softmax/sigmoid). The segmentation map 
produced is calculated by applying the final activation to the sum of all of these intermediate 
decoder feature maps upsampled to the same resolution as the final decoder feature map, so that 
each decoder block learns to make a direct contribution to the output class probabilities.

Note that other versions of deep supervision directly generate multiple output segmentation maps 
(at different spatial resolutions) and train directly on the ground truth mask resampled to 
match these. 

Input Pyramid Pooling
^^^^^^^^^^^^^^^^^^^^^

Input pyramid pooling is implemented as in the paper 
`A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (Abraham et al. 2019)`_.

Pyramid pooling can have slightly different meanings depending on the context. The version 
implemented here makes the input image available to the network encoder at multiple spatial resolutions 
through downsampling. This way each stage of the network encoder has access to both the 
feature maps of the previous encoder layer (which access the original image indirectly) 
and to the image itself directly, resampled to the spatial resolution of each encoder block.

Each intermediate encoder block is preceded by an additional set of convolutional filters 
which generate feature maps from the coarsened input image. These coarse feature 
maps are concatenated with the output of the previous encoder block and these together 
form the inputs of the encoder block.

.. figure:: media/ds_pp_sag.png
  :alt: Network architecture with ds + pp + sag

  Network architecture with input pyramid pooling, spatial attention gates and (another 
  variant of) deep supervision.

Empirically the improvements from pyramid pooling seem to be marginal with respect to most of the 
other architectural enhancements described, and the additional feature maps do make networks 
with these enabled heavier.

Grouped Convolutions
^^^^^^^^^^^^^^^^^^^^

Grouped convolutions are implemented according to the "ResNeXt" block described in the paper 
`Aggregated Residual Transformations for Deep Neural Networks (Xie et al. 2016)`_.

.. figure:: media/grouped_convs.png
  :alt: grouped convs

  Residual grouped convolution block (right) compared to regular residual block (left)

These are variations on a standard ResNet block which derive a set of *cardinality* low-dimensional 
projections of feature maps and apply 3x3 convolutions independently to these before re-concatenating 
them. While these worked well in image classification tasks, we've found empirically that they don't 
improve performance in segmentation.

Convolutional Block Attention Module (CBAM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CBAM is implemented according to the paper `CBAM\: Convolutional Block Attention Module (Woo et al. 2018)`_.

The CBAM module is in essence a simplified form of self-attention which enables a set of feature maps (from the 
output of a residual block) to calculate channel and spatial attention maps for reweighting themselves. 

.. figure:: media/cbam_module.png
  :alt: CBAM mod

  The channel and spatial attention modules making up the CBAM. The channel module passes a vector of 
  the max and average values of each channel across the whole spatial domain through an MLP and learns 
  a channel-wise reweighting with a sigmoid activation function. The spatial attention module performs 
  a global max and average pooling operation to derive two channel feature descriptors across the spatial 
  extent of the feature maps which are passed through a sigmoid convolution to derive a spatial attention 
  map. The kernel size of the final convolution is a hyperparameter which provides a degree of 
  context-awareness in the derivation of the attention values.

The channel and spatial attention maps reweight the feature maps sequentially, i.e. the input feature maps 
are first used to calculate channel attention maps, the channels are reweighted, and these reweighted 
feature maps are used to calculate spatial attention which reweights the whole set (spatially) one more time.

.. figure:: media/cbam_block.png
  :alt: cbam_block

  The CBAM block as positioned in a residual block; i.e. as a feature map postprocessing step.

These blocks are quite lightweight (due to the channel and spatial pooling operations) and the dynamic 
reweighting capability has been shown to lead to performance gains in various tasks. In experiments so 
far these yield a minor net benefit to segmentation performance.

Channel-Spatial Attention Gates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO: doesn't work as well as spatial attention in its current incarnation.

.. _segmentalist:

Segmentalist
------------

The :class:`~gim_cv.models.segmentalist.Segmentalist` model is a configurable 
fully convolutional residual encoder-decoder neural network with several optional
features. It can be created as follows::

    from gim_cv.models.segmentalist import Segmentalist

    # basic configuration with optional features off
    model = Segmentalist(
        layer_blocks=[2,2,2,2], # number of residual blocks per intermediate encoder/decoder stage
        last_decoder_layer_blocks=2, # number of residual blocks in the final stage
        initial_filters=64, # number of convolutional filters in first layer
        residual_filters=[64,128,256,512], # number  of residual filters per conv block per stage
        initial_kernel_size=(7,7), # convolution kernel size in the first layer
        head_kernel_size=(1,1), # final kernel size to produce segmentation map
        cardinality=1, # cardinality (ResNeXt grouped convolution parameter). 1 implies regular convs
        act='relu', # activation function
        downsample='pool', # downsampling method, 'pool' or 'strides'
        decoder_attention_gates=None, # None, 'SAG' (spatial attention gate) or 'CSAG' (channel + <-)
        encoder_cbam=False, # if True, enable CBAM module in encoder residual blocks
        decoder_cbam=False, # if True, enable CBAM module in decoder residual blocks
        pyramid_pooling=False, # if True, enable pyramid spatial pooling (multi-scale inputs)
        deep_supervision=False # if True, enable deep supervision
    )

In its basic configuration (as instantiated above) it's equivalent to a `DeepResUNet`_ model. Each of 
the various architectural enhancements above can be switched on independently. These can either modify 
the residual blocks themselves (in the case of grouped convolutions and the CBAM module), or they can 
add stages to the architecture itself (attention gates in the decoder, input pyramid pooling to the 
encoder block inputs, deep supervision to the decoder block outputs).

The :meth:`~gim_cv.models.segmentalist.Segmentalist.load_from_metadata` method allows one to construct 
a model from a row of a pandas dataframe containing the appropriate architecture initialisation 
parameters. See the API documentation for more details.


.. _`Semantic Segmentation of Urban Buildings from VHR Remote Sensing Imagery Using a Deep Convolutional Neural Network (Yi et al. 2019)` : `dru_paper`_
.. _`DeepResUNet` : `dru_paper`_
.. _`dru_paper`: https://www.mdpi.com/2072-4292/11/15/1774/htm







.. _Improving CT Image Tumor Segmentation Through Deep Supervision and Attentional Gates (Tureckova et al. 2020): https://www.frontiersin.org/articles/10.3389/frobt.2020.00106/full

.. _Aggregated Residual Transformations for Deep Neural Networks (Xie et al. 2016): http://arxiv.org/abs/1611.05431

.. _Attention U-Net\: Learning Where to Look for the Pancreas (Oktay et al. 2018): http://arxiv.org/abs/1804.03999

.. _CBAM\: Convolutional Block Attention Module (Woo et al. 2018): http://arxiv.org/abs/1807.06521

.. _A Novel Focal Tversky loss function with improved Attention U-Net for lesion segmentation (Abraham et al. 2019): http://arxiv.org/abs/1810.07842 
