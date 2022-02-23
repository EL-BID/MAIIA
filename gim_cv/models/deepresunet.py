""" deepresunet.py

    Object oriented layer-flexible implementation of the semantic segmentation
    architecture described in the publication:

    Semantic Segmentation of Urban Buildings from VHR Remote Sensing Imagery
    Using a Deep Convolutional Neural Network

    July 2019, Remote Sensing 11(15):1774
    
    `https://www.researchgate.net/publication/
    334743027_Semantic_Segmentation_of_Urban_Buildings_from_VHR_
    Remote_Sensing_Imagery_Using_a_Deep_Convolutional_Neural_Network`
"""
import tensorflow as tf
keras = tf.keras
layers = tf.keras.layers

import timbermafia as tm
from kerastuner import HyperModel

import gim_cv.losses as losses
import gim_cv.tools.keras_one_cycle_clr as clr

import logging

log = logging.getLogger(__name__)


def _derive_filters_list(output_filters:int, ratios:tuple=(1,2,2), ref_filter_index=-1):
    """ utility function for deriving lists of filter dimensions for resblocks
        given the number of output filters and ratios to fix the others based on
        it

        e.g. 128, (1, 2, 2) => 64, 128, 128
             32,  (1, 2, 4) => 8, 16, 32
    """
    resc_ratios = [f / ratios[ref_filter_index] for f in ratios]
    return [int(output_filters * r) for r in resc_ratios]


class BNActivation(tf.keras.Model, tm.Logged):#tf.keras.layers.Layer):
    """ Layer combining batch normalisation and activation
    """
    def __init__(self, name, act='relu', *args, **kwargs):
        """ 
        Parameters
        ----------
        name: str
            string name of layer in model
        act: str or callable
            activation fn to apply (None -> no activation)
        args: 
            passed to layer constructor
        kwargs:
            passed to layer constructor
        """
        super(BNActivation, self).__init__(name=name, *args, **kwargs)
        self.act = act
        self.bn = tf.keras.layers.BatchNormalization(name=f"{self.name}_bn")

    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        if self.act is not None:
            x = tf.keras.layers.Activation(self.act, name=f"{self.name}_act")(x)
        return x


class ConvBlock(tf.keras.Model, tm.Logged):#tf.keras.layers.Layer):
    """ Layer combining 3x3 2D convolution followed by batch norm and activation
    """
    def __init__(self,
                 name:str,
                 filters:int,
                 kernel_size:tuple=(3, 3),
                 padding:str="same",
                 strides:int=1,
                 act='relu',
                 *args, **kwargs):
        """ 
        Parameters
        ----------
        filters: int 
            number of convolutional filters
        name: str    
            name given to the layer
        padding: str
            defaults to 'same', s.t. output dims are same as input
        strides: int
            default to striding filters by one pixel
        """
        super(ConvBlock, self).__init__(name=name, *args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.act = act
        self.batch_norm_act = BNActivation(self.act)
        self.conv2d = tf.keras.layers.Conv2D(self.filters,
                                             self.kernel_size,
                                             padding=self.padding,
                                             strides=self.strides)

    def call(self, inputs, training=None):
        conv = self.conv2d(inputs)
        return self.batch_norm_act(conv, training=training)


class ResidualBlock(tf.keras.Model, tm.Logged):#tf.keras.layers.Layer):
    """ Residual block consisting of a pair of successive 3x3
        convolutions, followed by batch normalisation and activation and a 1x1
        convolution to reshape to the input size to get the residual value,
        before adding this to the input to make the final output.

        Pooling defines the downsampling ratio which occurs after the conv blocks
    """

    def __init__(self,
                 name:str,
                 output_filters:int,
                 kernel_size:tuple=(3, 3),
                 padding:str="same",
                 reshape_input:bool=False,
                 strides:int=1,
                 f_ratios:list=[1, 2, 2],
                 *args, **kwargs):
        """ 
        Parameters
        ----------
        x: 
            input tensor
        output_filters: 
            the number of filters to be output by the final convolutional layer
        kernel_size: 
            the convolutional kernel size
        padding: 
            specifies behaviour on if/how to pad the images
        strides: 
            the stride of the first convolution
        f_ratios: 
            ratios determining the number of filters in each
            conv layer, taken with reference to filters (the
            no. in the output layer). this will be normalised
            internally so that f_ratios[-1] == 1.0
            e.g. filters = 128, ratios = (1, 2, 2)
            => (64, 128, 128)
        reshape_input: 
            boolean to specify whether the output shape of the
            residual created by the conv blocks needs to be
            reshaped to match the input before addition
        """
        super(ResidualBlock, self).__init__(name=name, *args, **kwargs)
        self.output_filters = output_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.f_ratios = [f / max(f_ratios) for f in f_ratios]
        # calculate the number of filters per convolutional component
        self.filters = _derive_filters_list(self.output_filters, self.f_ratios)
        self.strides = strides
        # determine whether we need an extra (poss. strided) 1x1 convolution
        # to reshape the input volume to match the residual, if the block
        # performs downsampling by stride or has diff n_filters to input depth
        self.reshape_input = reshape_input
        # the main conv blocks to derive the residual
        self.conv_block_1 = ConvBlock(name=f"{self.name}_conv_2d_block_1",
                                      filters=self.filters[0],
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      strides=self.strides)
        self.conv_block_2 = ConvBlock(name=f"{self.name}_conv_2d_block_2",
                                      filters=self.filters[1],
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      strides=1)
        self.conv_2d_1x1 = tf.keras.layers.Conv2D(
            name=f"{self.name}_conv_2d_1x1",
            filters=self.filters[2],
            kernel_size=(1, 1),
            padding=self.padding,
            strides=1
        )
        # the blocks to situationally reshape the input to match the residual
        # this should apply strides and have the same number of filters as
        # the final 1x1 convolution
        if self.reshape_input:
            self.conv_2d_1x1_reshape = tf.keras.layers.Conv2D(
                name=f"{self.name}_conv_2d_1x1_reshape_input",
                filters=self.filters[2],
                kernel_size=(1, 1),
                padding=self.padding,
                strides=self.strides
            )
            self.bn = tf.keras.layers.BatchNormalization(
                name=f"{self.name}_bn_reshaped_input"
            )
            #log.debug(f"layer {self.name} using reshaping convolution in res block")
        else:
            pass
            #log.debug(f"layer {self.name} using identity mapping in res block")
        self.add = tf.keras.layers.Add()
        self.relu = tf.keras.layers.Activation('relu',
                                               name=f"{self.name}_relu")

    def call(self, inputs, training=None):
        # the input to which the residual will be added
        # if this doesn't need to be reshaped, it's just the original input
        inp = inputs
        # Derive residual value: (NxN conv, BN, act) x 2 -> 1x1 conv
        res = self.conv_block_1(inputs, training)
        res = self.conv_block_2(res, training)
        res = self.conv_2d_1x1(res)
        # if the convolutions changed the shape of the output volume so that
        # it no longer matches the input, we need to reshape the input
        if self.reshape_input:
            inp = self.conv_2d_1x1_reshape(inp)
            inp = self.bn(inp, training)
        # add the input to the residual value
        add = self.add([inp, res])
        # relu residual value
        return self.relu(add)

    def build(self, *args, **kwargs):
        #log.debug(f"{self.name} being built")
        super(ResidualBlock, self).build(*args, **kwargs)
        #self.built=True
        

class UpsampleConcatBlock(tf.keras.Model, tm.Logged):#tf.keras.layers.Layer):
    """ 
    Upsample input x and concatenate it with input from previous layers,
    then apply 1x1 convolution kernels to fix channel dimension to filters
    """
    def __init__(self,
                 name:str,
                 output_filters:int,
                 upsampling_size:tuple=(2,2),
                 *args, **kwargs):
        """ 
        Parameters
        ----------
        upsampling_size: (row,col) factors warning which to scale
                         feature maps
        filters: the number of filters to reshape the concatenated
                 feature maps to

        To avoid checkerboard artifacts, an alternative upsampling
        method is applied by classical upsampling followed by a regular
        convolution (that preserves the spatial dimensions).
        """
        super(UpsampleConcatBlock, self).__init__(name=name, *args, **kwargs)
        self.output_filters = output_filters
        self.upsampling_size = upsampling_size
        # simple upsampling
        self.upsample = tf.keras.layers.UpSampling2D(upsampling_size)
        self.concat = tf.keras.layers.Concatenate()
        # convolution to follow upsampling
        self.conv_2d_1x1 = tf.keras.layers.Conv2D(
            name=f"{self.name}_conv_2d_1x1",
            filters=self.output_filters,
            kernel_size=(1,1),
            padding="same"
        )

    def call(self, inputs, training=None):
        """ 
        Parameters
        ----------
        inputs : 
            (x, xskip) where x is input tensor of decoder convolutional feature maps and 
            xskip is tensor of feature maps with same dimensions from the downsampling stage
        """
        x, xskip = inputs
        # upsample either with a standard scaling or transpose convolutions
        u = self.upsample(x)
        c = self.concat([u, xskip])
        # need to have some kind of cropping or downsampling here if want any size
        # down-sampling filters not allowed when not integer mult
        # so when channel dims change, this fails
        return self.conv_2d_1x1(c)


class DownsampleBlock(tf.keras.Model, tm.Logged):
    def __init__(self,
                 name:str,
                 output_filters:int,
                 reshape_input:bool=False,
                 f_ratios:tuple=(1,2,2),
                 *args, **kwargs):
        """
        """
        super(DownsampleBlock, self).__init__(name=name, *args, **kwargs)
        self.output_filters = output_filters
        self.reshape_input = reshape_input
        self.f_ratios = f_ratios
        # maxpooling for downsample
        self.pool = tf.keras.layers.MaxPool2D(
            name=f"{self.name}_maxpool_2d_2x2",
            pool_size=(2,2),
            strides=2,
            padding="same"
        )
        # the first residual block should reshape the input if required, as the
        # number of filters here may increase wrt the previous layer
        name_blk_1 = f"{self.name}_res_block_1"
        self.res_block_1 = ResidualBlock(name=name_blk_1,
                                         output_filters=self.output_filters,
                                         strides=1,
                                         reshape_input=self.reshape_input,
                                         f_ratios=self.f_ratios)
        # by the second residual block the channels are fixed to the same as the
        # output of the first block so no reshaping necessary
        name_blk_2 = f"{self.name}_res_block_2"
        self.res_block_2 = ResidualBlock(name=name_blk_2,
                                         output_filters=self.output_filters,
                                         strides=1,
                                         reshape_input=False,
                                         f_ratios=self.f_ratios)
        # layer for adding input to residual and relu
        self.add = tf.keras.layers.Add(name=f"{self.name}_add")
        self.relu = tf.keras.layers.Activation('relu',
                                               name=f"{self.name}_relu")

    def call(self, inputs, training=None):
        downsampled_inputs = self.pool(inputs)
        res = self.res_block_1(downsampled_inputs, training)
        res = self.res_block_2(res, training)
        # NOTE that in the paper they calculate a residual of residuals and do
        # another addition here which i think is redundant, so will exclude
        #summed = self.add([downsampled_inputs, res])
        return res#self.relu(summed)


class UpsampleBlock(tf.keras.Model, tm.Logged):
    def __init__(self,
                 name:str,
                 upsample_output_filters:int,
                 resblock_output_filters:int,
                 f_ratios:tuple=(1,2,2),
                 *args, **kwargs):
        """ 
        Upsampling residual block, combining concatenated encoder layer
        output with upsampled output of previous decoder layer, followed by
        two residual blocks

        Parameters
        ----------
        upsample_output_filters:  
            The number of convolutional filters to be output by the UpsampleConcatBlock,
            and thus to be used as input to the residual blocks
        resblock_output_filters:  
            the number of convolutional filters to be output by the residual blocks
        """
        super(UpsampleBlock, self).__init__(name=name, *args, **kwargs)
        self.upsample_output_filters = upsample_output_filters
        self.resblock_output_filters = resblock_output_filters
        # if the number of
        self.reshape_input = (self.upsample_output_filters !=
                              self.resblock_output_filters)
        self.f_ratios = f_ratios
        # upsample prev decoder ouput and concatenate with encoder output skip
        usc_name = f"{self.name}_upsample_concat"
        self.upsample_concat = UpsampleConcatBlock(
            name=usc_name,
            output_filters=self.upsample_output_filters
        )
        # the first residual block should reshape the input if required, as the
        # number of filters here may increase wrt the previous layer
        name_blk_1 = f"{self.name}_res_block_1"
        self.res_block_1 = ResidualBlock(name=name_blk_1,
                                         output_filters=self.resblock_output_filters,
                                         strides=1,
                                         reshape_input=self.reshape_input,
                                         f_ratios=self.f_ratios)
        # by the second residual block the channels are fixed to the same as the
        # output of the first block so no reshaping necessary
        name_blk_2 = f"{self.name}_res_block_2"
        self.res_block_2 = ResidualBlock(name=name_blk_2,
                                         output_filters=self.resblock_output_filters,
                                         strides=1,
                                         reshape_input=False,
                                         f_ratios=self.f_ratios)

    def call(self, inputs, training=None):
        x, xskip = inputs
        # upsample the previous decoder output and concatenate with encoder skip
        x = self.upsample_concat([x, xskip])
        # pass through residual blocks
        x = self.res_block_1(x, training)
        return self.res_block_2(x, training)


## main model

class DeepResUNet(tf.keras.Model, tm.Logged):
    """
    tf.keras Model implementation of Deep Residual U-Net
    """
    def __init__(self,
                 name='DeepResUNet',
                 initial_conv_kernel:tuple=(5,5),
                 filters:list=[128,128,256,256,512],
                 res_block_ratios:tuple=(1,2,2),
                 input_image_shape:tuple=(None, None, 3),
                 downsample:str="pool",
                 *args, **kwargs):
        """
        Parameters:
            initial_conv_kernel: (int, int)
                The size of the first privileged conv kernel (5x5 in the paper, 7x7 in ResNet etc)
            filters: list of int
                The number of filters used from the first conn
            res_block_ratios: tuple of int 
                The ratios of the number of filters in each residual block (see ResBlock)
            input_image_shape: tuple of int 
                The dimensions of each image e.g. (224, 224, 3). Nones in the xy dims for variable.
            downsample: str
                "pool" or "stride" for maxpool/strided convolutions in first conv block

        """
        super(DeepResUNet, self).__init__(name=name, *args, **kwargs)
        self.filters = filters
        self.n_blocks = len(self.filters)
        self.downsample = downsample
        self.initial_conv_kernel = initial_conv_kernel
        self.res_block_ratios = res_block_ratios
        # -- encoder blocks
        # initial large-kernel convolution + downsampling
        ci_name = f"conv_2d_initial"
        self.conv_2d_initial = tf.keras.layers.Conv2D(
            input_shape=(None, None, 3),
            name=ci_name,
            filters=self.filters[0],
            kernel_size=self.initial_conv_kernel,
            padding="same",
            strides=1
        )
        # initialise the different encoder and decoder blocks
        self.downsample_blocks, self.upsample_blocks = [], []
        for ix in range(1, self.n_blocks):
            # ---- downsample blocks
            # each consists of pooling, 2 x res blocks, then input addition + relu
            # at each stage, if the number of channels coming in from the previous
            # layer is different from that coming out of this layer, we need to
            # tell the residual blocks to use 1x1 conv filters to reshape input
            ds_name = f"downsample_block_{ix}"
            in_channels, out_channels = self.filters[ix-1], self.filters[ix]
            # determine if the block transition changes the number of filters
            reshape_input = (in_channels != out_channels)
            ds_blk = DownsampleBlock(name=ds_name,
                                     output_filters=out_channels,
                                     reshape_input=reshape_input,
                                     f_ratios=self.res_block_ratios)
            setattr(self, ds_name, ds_blk)
            self.downsample_blocks.append(ds_blk)
            # ---- upsample blocks
            # each consists of upsampling the previous decoder layer, followed
            # by concatenation with the corresponingly sized encoder layer's
            # feature maps, 1x1 conv to half channels then two resblocks
            us_name = f"upsample_block_{ix}"
            # the upsampling blocks take input from the immediately preceding
            # upsampling block, which is upsampled and concatenated with the
            # downsampled outputs from the encoder blocks with the same-sized
            # feature maps.
            # the last encoder output is the first decdoder input.
            in_channels = self.filters[self.n_blocks-ix]
            out_channels = self.filters[self.n_blocks-ix-1]
            # the upsampling block first reshapes the concatenated upsampled
            # input to input filters
            us_blk = UpsampleBlock(name=us_name,
                                   upsample_output_filters=in_channels,
                                   resblock_output_filters=out_channels,
                                   f_ratios=self.res_block_ratios)
            setattr(self, us_name, us_blk)
            self.upsample_blocks.append(us_blk)

        # index pairings for matching blocks
        self.block_pairings = list(zip(
            range(1, self.n_blocks),range(self.n_blocks, 0, -1)))
        # output
        self.output_layer = tf.keras.layers.Conv2D(1, (1, 1),
                                                   name=f"{self.name}_output",
                                                   padding="same",
                                                   activation="sigmoid")
        self.checkpoint_uuid = None
        
    def call(self, inputs, training=None):
        # track encoder layer outputs from beginning
        encoder_outputs = [inputs]
        # initial large-kernel convolution
        encoder_outputs.append(
            self.conv_2d_initial(encoder_outputs[0])
        )
        # ------------------------------- residual downsampling blocks (encoder)
        # calculate downsampling block outputs sequentially
        for i in range(1, self.n_blocks):
            encoder_outputs.append(
                self.downsample_blocks[i-1](encoder_outputs[i])
            )
        # --------- residual upsampling blocks with concatenated skips (decoder)
        # pass last encoder output (following the "bridge" convolutions) to the
        # first decoder layer, pretending these are the first decoder outputs
        # for brevity
        decoder_outputs = [encoder_outputs[-1]]
        for i in range(self.n_blocks-1):
            # each layer of decoder takes the previous layer's output along with
            # the matching encoder layer's output
            upsampled = self.upsample_blocks[i](
                [decoder_outputs[i], encoder_outputs[self.n_blocks-i-1]]
            )
            decoder_outputs.append(upsampled)
        # output layer (sigmoid)
        return self.output_layer(decoder_outputs[-1])

    @property
    def checkpoint_uuid(self):
        return self._checkpoint_uuid
    
    @checkpoint_uuid.setter
    def checkpoint_uuid(self, value):
        self.log.debug(f"Setting checkpoint uuid to: {value}")
        self._checkpoint_uuid = value
    
    @classmethod
    def load_from_metadata(cls,
                           row,
                           val=True,
                            **model_kwargs):
        """
        Loads a DeepResUNet with the weights loaded from the checkpoint with 
        the lowest validation loss.

        If val is False, uses the checkpoint with the lowest training loss.
        """
        initial_conv_kernel = (int(row.kernel_size), int(row.kernel_size))
        filters = row.filters.split(',')
        model = DeepResUNet(
            initial_conv_kernel=initial_conv_kernel,
            filters=[int(f) for f in filters],
            **model_kwargs
        )
        model.checkpoint_uuid = row.uuid4
        # use same setup as training script to ensure parity in loaded model
        opt = tf.keras.optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False
        )
        # select metrics
        metrics = [
            losses.tversky_index,
            losses.jaccard_index,
            losses.recall,
            losses.precision,
            losses.specificity,
            losses.npv,
            losses.dice_coefficient
        ]
        # TODO - implement loss function args if necessary
        loss_fn = getattr(losses, row.loss_fn)
        model.compile(optimizer=opt,
                      loss=loss_fn,
                      metrics=metrics)
        cp = row.lowest_val_loss_ckpt if val else row.lowest_loss_ckpt
        if cp:
            model.load_weights(cp)
            return model
        else:
            raise ValueError("No checkpoint found in directory!")
    

class HyperDeepResUNet(HyperModel, tm.Logged):
    """ 
    kerastuner.Hypermodel for DeepResUNet
    
    very helpful:
    https://keras-team.github.io/keras-tuner/examples/helloworld/
    """
    def __init__(self,
                 loss,
                 lf_args=None,
                 ocp=True,
                 metrics=[],
                 callbacks=[],
                 momentum_range=(0.95, 0.85)):
        # get loss function and derive name
        # if it takes additional arguments, supply them
        self.loss_fn = getattr(losses, loss)
        if lf_args:
            self.loss_fn = self.loss_fn(**lf_args)
        self.metrics = list(metrics)
        self.momentum_range = momentum_range

    def build(self, hp):
        # -- hyperparameters
        # initial conv kernel size
        init_conv_kern_size = hp.Choice('init_conv_kern_size', values=[5,7], default=5)
        # layers and filters - can specify in __init__ to override scan
        n_layers = hp.Int('n_layers', 5, 8, default=5)
        # set default filters to [128, 128, 256, 256, 256]
        filters = ([hp.Choice(f'filters_{i}',
                             values=[32, 64, 128, 256],
                             default=128)
                    for i in range(2)] +
                   [hp.Choice(f'filters_{i}',
                              values=[32, 64, 128, 256],
                              default=256)
                    for i in range(2,5)] +
                   [hp.Choice(f'filters_{i}',
                              values=[128, 256, 512],
                              default=256)
                    for i in range(5, n_layers)])
        # learning rate(s)
        lr_init = hp.Float('lr_init', min_value=1e-5, max_value=1e-1,
                          sampling='log', default=5e-3)
        if hp.Boolean('use_ocp', default=False):
            lr_mult = hp.Choice('lr_multiplier', values=[5, 10], default=10)
        # gradient descent optimiser
        opt = hp.Choice('opt', values=['sgd', 'adam'], default='sgd')
        if opt == 'sgd':
            opt = tf.keras.optimizers.SGD(
                learning_rate=lr_init,
                momentum=self.momentum_range[0],
                nesterov=False
            )
        elif opt == 'adam':
            opt = tf.keras.optimizers.Adam(
                learning_rate=lr_init,
                beta_1=0.9,
                beta_2=0.999,
                amsgrad=False
            ) # check out RADAM?
        else:
            raise ValueError(f"Optimiser {opt} not understood")
        # build and compile the model
        model = DeepResUNet(initial_conv_kernel=(init_conv_kern_size,
            init_conv_kern_size),
                            filters=filters)
        model.build(input_shape=(None, None, None, 3))
        model.compile(optimizer=opt,
                      loss=self.loss_fn,
                      metrics=self.metrics)
        return model
