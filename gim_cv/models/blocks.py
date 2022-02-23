from typing import Union, Optional, List, Tuple

import timbermafia as tm
import tensorflow as tf
from tensorflow import einsum
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Softmax, Conv3D
from einops import rearrange
keras = tf.keras
layers = tf.keras.layers
K = keras.backend

# when debugging, it's useful to have all layers be Model subclasses to have access to .summary and .fit methods
DEBUG_COMPONENTS = True

if DEBUG_COMPONENTS:
    layer_cls = keras.Model
else:
    layer_cls = layers.Layer

class BNActivation(layer_cls, tm.Logged):
    """Layer combining batch normalisation and activation
    
    Parameters
    ----------
    name : str
        string name of layer in model
    act : str or callable
        activation fn to apply (None -> no activation)
    axis : int, optional
        axis over which to perform batch normalisation
    args : 
        passed to layer constructor
    kwargs :
        passed to layer constructor
    """
    def __init__(self, name, act='relu', axis=-1, *args, **kwargs):
        super(BNActivation, self).__init__(name=name, *args, **kwargs)
        self.act = act
        self.bn = layers.BatchNormalization(name=f"{self.name}_bn")
        self.axis = axis
        
    def build(self, input_shape):
        self.act = layers.Activation(self.act, name=f"{self.name}_act")
        self.bn = layers.BatchNormalization(name=f"{self.name}_bn", axis=self.axis)
    
    def call(self, inputs, training=None):
        x = self.bn(inputs, training)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvBlock(layer_cls, tm.Logged):
    """ Layer combining 3x3 2D convolution followed by batch norm and activation
    
    Parameters
    ----------
    filters : int 
        number of convolutional filters
    name : str    
        name given to the layer
    padding : str
        defaults to 'same', s.t. output dims are same as input
    strides : int
        default to striding filters by one pixel
    axis : int, optional
        axis over which to perform batch normalisation
    """
    def __init__(self,
                 name:str,
                 filters:int,
                 kernel_size:tuple=(3, 3),
                 in_shape:Optional[Tuple[int]]=None,
                 padding:str="same",
                 strides:int=1,
                 act='relu',
                 axis=-1,
                 *args, **kwargs):
        super(ConvBlock, self).__init__(name=name, *args, **kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.in_shape = in_shape
        self.padding = padding
        self.strides = strides
        self.act = act
        self.axis = axis  
        
    def build(self, input_shape):
        self.batch_norm_act = BNActivation(self.act, axis=self.axis)
        kwargs = {}
        #if self.in_shape is not None:
        #    kwargs.update(input_shape=self.in_shape)
        self.conv2d = layers.Conv2D(
            self.filters,
            self.kernel_size,
            input_shape=input_shape,
            padding=self.padding,
            strides=self.strides,
            **kwargs
        )
        
    def call(self, inputs, training=None):
        conv = self.conv2d(inputs)
        return self.batch_norm_act(conv, training=training)
        

class GroupedConvBlock(layer_cls):
    """
    Grouped Convolutions as in the ResNeXt paper
    (see https://arxiv.org/pdf/1611.05431.pdf)

    Given a feature map with some number of input channels, divide these
    channels up into N=`cardinality` groups of `grouped_channels` channels.
    Each subset of the input channels are subjected to independent convolutions,
    and the output feature maps are concatenated before batch norm and activation.

    Reimplementation of:
    https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py

    Parameters
    ----------
    name :
        The name to be assigned to this layer instance
    grouped_channels :
        The total number of filters per group: int(total_filters / cardinality)
    cardinality :
        The number of groups of filters, each with grouped_channels channels
    strides : 
        The stride length of the convolution. Downsamples if > 1.
    weight_decay :
        L2 regularisation penality
    act :
        String name of activation function
    axis :
        Axis over which to perform batch normalisation
    """
    def __init__(
        self,
        name:str,
        grouped_channels:int,
        cardinality:int,
        strides:int,
        weight_decay:Optional[float]=None,#5e-4,
        act:str='relu',
        axis:int=-1,
        use_bias=False,
        *args,
        **kwargs
    ):
        super(GroupedConvBlock, self).__init__(name=name, *args, **kwargs)
        self.grouped_channels = grouped_channels
        self.cardinality = cardinality
        self.strides = strides
        self.weight_decay = weight_decay
        self.axis = axis
        self.act = act
        self.reg = keras.regularizers.l2(self.weight_decay) if self.weight_decay else None
        self.use_bias = use_bias
        
    def build(self, input_shape):
        # with cardinality 1, it is a standard convolution and grouped_channels == total_channels
        if self.cardinality == 1:
            self.reg_conv = layers.Conv2D(
                self.grouped_channels,
                kernel_size=(3, 3),
                padding='same',
                use_bias=self.use_bias,
                strides=(self.strides, self.strides),
                kernel_initializer='he_normal',
                kernel_regularizer=self.reg
            )
        # otherwise we split the channel dimension up into a factor of cardinality subsets
        else:
            self.channel_sel_lambdas = []
            self.grouped_convs = []
            for c in range(self.cardinality):
                self.channel_sel_lambdas.append(
                    layers.Lambda(
                        lambda z: z[:, :, :, c * self.grouped_channels:((c + 1) * self.grouped_channels)]
                    )
                )
                # different convs for each subset of channels 
                self.grouped_convs.append(
                    layers.Conv2D(
                        self.grouped_channels,
                        kernel_size=(3, 3),
                        padding='same',
                        use_bias=self.use_bias,
                        strides=(self.strides, self.strides),
                        kernel_initializer='he_normal',
                        kernel_regularizer=self.reg
                    )
                )
            self.concat = layers.Concatenate(axis=-1)
        self.bn_act = BNActivation(name=self.name, act=self.act, axis=self.axis)
    
    def call(self, inputs, training=None):
        initial = inputs
        group_list = []
        
        # regular conv
        if self.cardinality == 1:
            x = self.reg_conv(inputs)
        # grouped convs
        else:
            for ix in range(len(self.channel_sel_lambdas)):
                x = self.channel_sel_lambdas[ix](inputs)
                x = self.grouped_convs[ix](x)
                group_list.append(x)
            x = self.concat(group_list)
        x = self.bn_act(x, training=training)
        return x
    
    
class LambdaLayer(layer_cls):
    """
    Lambda layer as described in the paper:
    
    LambdaNetworks: Modeling Long-Range Interactions Without Attention
    https://openreview.net/pdf?id=xTJEN-ggl1b
    
    Implementation: https://github.com/seanmor5/tf_lambda
    
    Parameters
    ----------
    dim_k : 
        
    n : 
    
    r : 
    
    heads : 
    
    dim_out : 
    
    dim_u : 
    
    strides : 
    
    pool_size : 
    
    """
    def __init__(
        self,
        *,
        dim_k,
        n = None,
        r = None,
        heads = 4,
        dim_out = None,
        dim_u = 1,
        strides=1,
        pool_size=(3,3)
    ):
        super(LambdaLayer, self).__init__()
        self.dim_k = dim_k
        self.n = n
        self.r = r
        self.dim_out = dim_out
        self.dim_u = dim_u
        self.heads = heads
        self.strides = strides
        self.pool_size = pool_size
        assert (self.dim_out % self.heads) == 0, f"{self.dim_out} output fmaps don't divide by {self.heads} heads"
        self.dim_v = self.dim_out // self.heads
        self.local_contexts = self.r is not None
    
    def build(self, input_shape):
        self.b, self.hh, self.ww, self.c = input_shape
        self.to_q = layers.Conv2D(filters = self.dim_k * self.heads, kernel_size = (1, 1), use_bias=False)
        self.to_k = layers.Conv2D(filters = self.dim_k * self.dim_u, kernel_size = (1, 1), use_bias=False)
        self.to_v = layers.Conv2D(filters = self.dim_v * self.dim_u, kernel_size = (1, 1), use_bias=False)

        self.norm_q = layers.BatchNormalization()
        self.norm_v = layers.BatchNormalization()

        if self.local_contexts:
            assert (self.r % 2) == 1, 'Receptive kernel size should be odd.'
            self.pad_fn = lambda x: tf.pad(x, tf.constant([[0, 0], [self.r // 2, self.r // 2], [self.r // 2, self.r // 2], [0, 0]]))
            self.pos_conv = layers.Conv3D(filters = self.dim_k, kernel_size = (1, self.r, self.r))
            self.flatten = layers.Flatten()
        else:
            assert self.n is not None, 'You must specify the total sequence length (h x w)'
            self.pos_emb = self.add_weight(name='position_embed', shape=(self.n, self.n, self.dim_k, self.dim_u))
        
        if self.strides > 1:
            self.avgpool = layers.AveragePooling2D(pool_size=self.pool_size, strides=self.strides, padding="same")
            
    def call(self, x):
        # For verbosity and understandings sake
        b, hh, ww, c, u, h = self.b, self.hh, self.ww, self.c, self.dim_u, self.heads

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        q = self.norm_q(q)
        v = self.norm_v(v)

        q = rearrange(q, 'b hh ww (h k) -> b h (hh ww) k', h = h)
        k = rearrange(k, 'b hh ww (k u) -> b u (hh ww) k', u = u)
        v = rearrange(v, 'b hh ww (v u) -> b u (hh ww) v', u = u)

        k = tf.nn.softmax(k, axis=-1)

        lambda_c = einsum('b u m k, b u m v -> b k v', k, v)
        Y_c = einsum('b h n k, b k v -> b n h v', q, lambda_c)

        if self.local_contexts:
            v = rearrange(v, 'b u (hh ww) v -> b v hh ww u', hh = hh, ww = ww)
            # We need to add explicit padding across the batch dimension
            lambda_p = tf.map_fn(self.pad_fn, v)
            lambda_p = self.pos_conv(lambda_p)
            lambda_p = tf.reshape(
                lambda_p,
                (tf.shape(lambda_p)[0], tf.shape(lambda_p)[1], tf.shape(lambda_p)[2] * tf.shape(lambda_p)[3], tf.shape(lambda_p)[4]))
            Y_p = einsum('b h n k, b v n k -> b n h v', q, lambda_p)
        else:
            lambda_p = einsum('n m k u, b u m v -> b n k v', self.pos_emb, v)
            Y_p = einsum('b h n k, b n k v -> b n h v', q, lambda_p)

        Y = Y_c + Y_p
        out = rearrange(Y, 'b (hh ww) h v -> b hh ww (h v)', hh = hh, ww = ww)
        if self.strides > 1:
            out = self.avgpool(out)
        return out


class LambdaConv(layer_cls):
    """
    Wrapper for Lambda layer with fixed receptive field "r" that looks like a convolution.
    """
    def __init__(self, channels_out, *, receptive_field = 23, key_dim = 16, intra_depth_dim = 1, heads = 4, strides=1):
        super(LambdaConv, self).__init__()
        self.channels_out = channels_out
        self.receptive_field = receptive_field
        self.key_dim = key_dim
        self.intra_depth_dim = intra_depth_dim
        self.heads = heads
        self.strides = strides

    def build(self, input_shape):
        self.layer = LambdaLayer(
            dim_out = self.channels_out,
            dim_k = self.key_dim,
            heads = self.heads,
            r = self.receptive_field,
            n = input_shape[1] * input_shape[2],
            strides=self.strides
        )

    def call(self, x):
        return self.layer(x)

    
class LambdaConvBlock(layer_cls, tm.Logged):
    """ Layer combining 3x3 2D convolution followed by batch norm and activation
    
    Parameters
    ----------
    filters : int 
        number of convolutional filters
    name : str    
        name given to the layer
    padding : str
        defaults to 'same', s.t. output dims are same as input
    strides : int
        default to striding filters by one pixel
    axis : int, optional
        axis over which to perform batch normalisation
    """
    def __init__(self,
                 channels_out,
                 receptive_field = 23,
                 key_dim = 16,
                 intra_depth_dim = 1,
                 heads = 4,
                 strides = 1,
                 act='relu',
                 axis=-1,
                 *args, **kwargs):
        super(LambdaConvBlock, self).__init__(*args, **kwargs)
        self.channels_out = channels_out
        self.receptive_field = receptive_field
        self.key_dim = key_dim
        self.intra_depth_dim = intra_depth_dim
        self.heads = heads
        self.strides = strides
        self.act = act
        self.axis = axis

    def build(self, input_shape):
        self.batch_norm_act = BNActivation(self.act, axis=self.axis)
        self.lambda_conv = LambdaConv(
            channels_out = self.channels_out,
            receptive_field = self.receptive_field,
            key_dim = self.key_dim,
            intra_depth_dim = self.intra_depth_dim,
            heads = self.heads,
            strides = self.strides
        )
        
    def call(self, inputs, training=None):
        conv = self.lambda_conv(inputs)
        return self.batch_norm_act(conv, training=training)
    
    
def _derive_filters_list(output_filters:int, ratios:tuple=(1,2,2), ref_filter_index=-1):
    """ utility function for deriving lists of filter dimensions for resblocks
        given the number of output filters and ratios to fix the others based on
        it

        e.g. 128, (1, 2, 2) => 64, 128, 128
             32,  (1, 2, 4) => 8, 16, 32
    """
    resc_ratios = [f / ratios[ref_filter_index] for f in ratios]
    return [int(output_filters * r) for r in resc_ratios]


class ResidualBlock(layer_cls):
    """
    A flexible residual block.

    Uses a ResNet-style bottleneck convolution sequence to squeeze the input channels
    down to filters, perform the main 3x3 convolution then expand the output channels
    again to 2*filters.

    If lambda_conv is True, uses a Lambda Convolution instead of the main 3x3 conv.
    
    See the paper: https://openreview.net/pdf?id=xTJEN-ggl1b
    
    If False, uses a ResNeXt-style grouped convolution block for the main 3x3 conv.

    In this case, first 1x1 convolution block effectively derives a bunch of low
    -dimensional embeddings of input feature maps. Each of these is 
    passed into a grouped convolution block and processed independently, 
    before concatenation and a final 1x1 convolution to control the number
    of output channels.

    See the paper: https://arxiv.org/pdf/1611.05431.pdf

    Reimplementation of:
    https://github.com/titu1994/Keras-ResNeXt/blob/master/resnext.py

    Parameters
    ----------
    filters :
        The number of filters in the main convolution (which is either a 3x3 grouped 
        ResNeXt-style convolution or a Lambda convolution). Output of the block is 
        expanded by a kernel size 1 convolution to 2*filters channels.
    cardinality :
        The cardinality (number of groups) of the grouped convolution. Ignored if 
        lambda_conv == True and a lambda convolution is used instead of a ResNeXt one.
    strides :
        The strides used to downsample (if any - default 1 and no downsampling)
    lambda_conv :
        Boolean flag to enable experimental lambda convolution instead of ResNeXt conv
        for the main convolution.
    act :
        String tag for activation function
    channel_expansion_factor : 
        Integer multiplier for output channels
    """
    def __init__(
        self,
        filters:int=64,
        cardinality:int=8,
        strides:int=1,
        lambda_conv:bool=False,
        act='relu',
        channel_expansion_factor:int=2,
        *args,
        **kwargs
    ):
        super(ResidualBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.cardinality = cardinality
        self.strides = strides
        self.lambda_conv = lambda_conv
        #self.weight_decay = weight_decay
        self.grouped_channels = int(self.filters / self.cardinality)
        self.act = act
        self.channel_expansion_factor = channel_expansion_factor

    def build(self, input_shape):
        # if the input doesn't have the same number of filters as the output, use a 
        # kernel-size 1 conv to match the feature map dimensions
        if (input_shape[-1] != self.channel_expansion_factor * self.filters) or (self.strides != 1):
            self.init_conv = layers.Conv2D(
                self.filters * self.channel_expansion_factor,
                kernel_size=(1, 1),
                padding='same',
                strides=(self.strides, self.strides),
                use_bias=False,
                kernel_initializer='he_normal',
                #kernel_regularizer=self.reg
            )
            self.init_bn = layers.BatchNormalization(axis=-1)
        # first 2D convolution: define linear projections onto which grouped convs act
        self.conv2d_1 = layers.Conv2D(
            filters=self.filters,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            #kernel_regularizer=self.reg
        )
        self.bn_act = BNActivation(name=self.name+'_BNAct', act=self.act)
        # main conv block: regular or grouped convolutions if lambda_conv = False,
        # otherwise Lambda convolutions
        if self.lambda_conv:
            #if self.strides != 1:
            #    raise NotImplementedError("Lambda convolution with strides isn't defined")
            self.main_conv = LambdaConvBlock(
                name=self.name + '_LambdaConvBlock',
                channels_out=self.filters,
                strides=self.strides,
                act=self.act
            )
        else:
            self.main_conv = GroupedConvBlock(
                name=self.name + '_GroupedConvBlock',
                grouped_channels=self.grouped_channels,
                cardinality=self.cardinality,
                strides=self.strides,
                #weight_decay=self.reg,
                act=self.act,
            )
        # final conv: expand output channels
        self.conv2d_2 = layers.Conv2D(
            filters=self.filters * self.channel_expansion_factor,
            kernel_size=(1, 1),
            padding='same',
            use_bias=False,
            kernel_initializer='he_normal',
            #kernel_regularizer=self.reg
        )
        self.bn = layers.BatchNormalization(axis=-1)
        self.add = layers.Add()
        self.act_out = layers.Activation(self.act)
        
    def call(self, inputs, training=None):
        init = inputs
        # make sure input channels match output for residual connection to work
        if (init.shape[-1] != self.channel_expansion_factor * self.filters) or (self.strides != 1):
            init = self.init_conv(init)
            init = self.init_bn(init)
        x = self.conv2d_1(inputs)
        x = self.bn_act(x, training=training)
        x = self.main_conv(x, training=training)
        x = self.conv2d_2(x)
        x = self.bn(x, training=training)
        x = self.add([init, x])
        x = self.act_out(x)
        return x

    
class DRUNetResidualBlock(layer_cls, tm.Logged):#tf.keras.layers.Layer):
    """ Variation of residual block used in the DeepResUNet
    
        Residual block consisting of a pair of successive 3x3
        convolutions, followed by batch normalisation and activation and a 1x1
        convolution to reshape to the input size to get the residual value,
        before adding this to the input to make the final output.
    """

    def __init__(self,
                 name:str,
                 filters:int,
                 cardinality:int=1,
                 kernel_size:tuple=(3, 3),
                 padding:str="same",
                 cbam:bool=False,
                 lambda_conv:bool=False,
                 strides:int=1,
                 f_ratios:list=[1, 2, 2],
                 act='relu',
                 *args, **kwargs):
        """ 
        Parameters
        ----------
        x: 
            input tensor
        filters: 
            the number of filters in the main 3x3 convolution (->filters[1])
        cardinality:
            the cardinality of the main convolution (== 1 => regular 3x3 conv)
        kernel_size: 
            the convolutional kernel size
        padding: 
            specifies behaviour on if/how to pad the images
        cbam:
            enable convolutional block attention module
        lambda_conv:
            replace the regular 3x3 convolutions with lambda convolutions (experimental)
        strides: 
            the stride of the first convolution
        f_ratios: 
            ratios determining the number of filters in each
            conv layer, taken with reference to filters (the
            no. in the "main" middle conv). this will be normalised
            internally so that f_ratios[1] == 1.0
            e.g. filters = 128, ratios = (1, 2, 2)
            => (64, 128, 128)
        """
        super(DRUNetResidualBlock, self).__init__(name=name, *args, **kwargs)
        self.filters_ = filters
        self.kernel_size = kernel_size
        self.cardinality = cardinality
        self.padding = padding
        self.cbam = cbam
        self.lambda_conv = lambda_conv
        self.f_ratios = [f / max(f_ratios) for f in f_ratios]
        self.act = act
        # calculate the number of filters per convolutional component using the central 3x3 conv as reference
        self.filters = _derive_filters_list(self.filters_, ref_filter_index=1, ratios=self.f_ratios)
        self.strides = strides
        
    def build(self, input_shape):
        # determine whether we need an extra (poss. strided) 1x1 convolution
        # to reshape the input volume to match the residual, if the block
        # performs downsampling by stride or has diff n_filters to input depth
        self.reshape_input = (input_shape[-1] != self.filters[-1])
        #print(self.name, f"is reshaping the input - input filters: {input_shape[-1]}, output filters: {self.filters[-1]}")
        #print(self.reshape_input)
        # the main conv blocks to derive the residual
        self.conv_block_1 = ConvBlock(name=f"{self.name}_conv_2d_block_1",
                                      filters=self.filters[0],
                                      act=self.act,
                                      kernel_size=self.kernel_size,
                                      padding=self.padding,
                                      strides=self.strides)
        # main conv block: regular or grouped convolutions if lambda_conv = False,
        # otherwise Lambda convolutions
        if self.lambda_conv:
            #if self.strides != 1:
            #    raise NotImplementedError("Lambda convolution with strides isn't defined")
            self.conv_block_2 = LambdaConvBlock(
                name=f"{self.name}_lambda_conv_2d_block_2",
                channels_out=self.filters[1],
                strides=self.strides
            )
        else:
            self.conv_block_2 = GroupedConvBlock(
                name=f"{self.name}_conv_2d_block_2",
                grouped_channels=int(self.filters[1]/self.cardinality),#grouped_channels,
                cardinality=self.cardinality,
                strides=self.strides,
                #weight_decay=self.reg,
                act=self.act,
            )
        self.conv_2d_1x1 = layers.Conv2D(
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
            self.conv_2d_1x1_reshape = layers.Conv2D(
                name=f"{self.name}_conv_2d_1x1_reshape_input",
                filters=self.filters[2],
                kernel_size=(1, 1),
                padding=self.padding,
                strides=self.strides
            )
            self.bn = layers.BatchNormalization(
                name=f"{self.name}_bn_reshaped_input"
            )
            #log.debug(f"layer {self.name} using reshaping convolution in res block")
        else:
            pass
            #log.debug(f"layer {self.name} using identity mapping in res block")
        # create a CBAM block if enabled
        if self.cbam:
            self.cbam_block = CBAM(reduction_ratio=8, kernel_size=(7,7))
        self.add = layers.Add()
        self.relu = layers.Activation(self.act, name=f"{self.name}_act")

    def call(self, inputs, training=None):
        #print(self.name, "call")
        #print("reshape input is:", self.reshape_input)
        # the input to which the residual will be added
        # if this doesn't need to be reshaped, it's just the original input
        inp = inputs
        #print("input shape:", inputs.shape)
        # Derive residual value: (NxN conv, BN, act) x 2 -> 1x1 conv
        res = self.conv_block_1(inputs, training)
        res = self.conv_block_2(res, training)
        res = self.conv_2d_1x1(res)
        # if the convolutions changed the shape of the output volume so that
        # it no longer matches the input, we need to reshape the input
        if self.reshape_input:
            #print("I am reshaping input")
            inp = self.conv_2d_1x1_reshape(inp)
            inp = self.bn(inp, training)
        # apply CBAM to reweight feature maps with spatial and channel attention
        if self.cbam:
            res = self.cbam_block(res)
        #print("residual shape:", res.shape)
        # add the input to the residual value
        add = self.add([inp, res])
        # relu residual value
        return self.relu(add)
    
    
class UpsampleConcatBlock(layer_cls, tm.Logged):#tf.keras.layers.Layer):
    """ 
    Upsamples input x and concatenates it with input from previous layers,
    [NOT] then apply 1x1 convolution kernels to fix channel dimension to filters
    [our residual block will do this for us]
        
    Parameters
    ----------
    output_filters : 
        the number of filters to reshape the concatenated feature maps to
    upsampling_size : optional
        (row,col) factors by which to scale feature maps
    interpolation : optional
        The type of interpolation to use ('bilinear' or 'nearest')

    To avoid transpose conv checkerboard artifacts, an we use this
    upsampling followed by a regular convolution (that preserves the spatial dimensions) 
    to do learnable upsampling.         
    """
    def __init__(self,
                 name:str,
                 #output_filters:int,
                 upsampling_size:tuple=(2,2),
                 interpolation:str='bilinear',
                 *args, **kwargs):
        super(UpsampleConcatBlock, self).__init__(name=name, *args, **kwargs)
        #self.output_filters = output_filters
        self.upsampling_size = upsampling_size
        self.interpolation = interpolation

    def build(self, input_shape):
        # simple upsampling
        self.upsample = layers.UpSampling2D(self.upsampling_size, interpolation=self.interpolation)
        self.concat = layers.Concatenate()
        # convolution to follow upsampling
        #self.conv_2d_1x1 = layers.Conv2D(
        #    name=f"{self.name}_conv_2d_1x1",
        #    filters=self.output_filters,
        #    kernel_size=(1,1),
        #    padding="same"
        #)

    def call(self, inputs, training=None):
        """ 
        Parameters
        ----------
        inputs : 
            (x, xskip), where x is feature maps from the 
            decoder which will be upsampled to match xskip spatially. 
            xskip is tensor of feature maps from the encoder stage skip line 
        """
        x, xskip = inputs
        # upsample either with a standard scaling or transpose convolutions
        #print(f"upsampling decoder input: {x.shape}")
        u = self.upsample(x)
        #print(f"concatenating upsampled decoder input {u.shape} with skip {xskip.shape}")
        c = self.concat([u, xskip])
        return c#self.conv_2d_1x1(c)
    
    
class EncoderBlock(layer_cls, tm.Logged):
    """
    A block representing a layer of the convnet where convolutions generate
    feature maps of fixed size with a fixed number of convolutional filters.
    
    Composed of multiple residual blocks, with batch normalisation and activation.
    
    Parameters
    ----------
    filters : 
        The number of filters in the main convolution. 
        Block output is then filters channels thick.
    n_res_blocks : 
        The number of residual blocks in the encoder block.
    cardinality : 
        The number of groups in ResNeXt grouped convolution.
    lambda_conv : 
        Boolean flag switching on Lambda convolutions inside the residual blocks
    act : 
        String name of activation function to use.
    cbam:
        Enables convolutional block attention module in the residual blocks
    args : 
        passed to Layer constructor
    kwargs :
        passed to Layer constructor
    """
    def __init__(
        self,
        filters:int,
        n_res_blocks:int=2,
        cardinality:int=1,
        strides:int=1,
        lambda_conv:bool=False,
        act:str='relu',
        cbam:bool=False,
        #channel_expansion_factor:int=2,
        *args, **kwargs
    ):
        super(EncoderBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.n_res_blocks = n_res_blocks
        self.cardinality = cardinality
        self.strides = strides
        self.lambda_conv = lambda_conv
        self.act = act
        self.cbam = cbam
        #self.channel_expansion_factor = channel_expansion_factor

    def build(self, input_shape):    
        self.res_blocks = []
        for blk_ix in range(self.n_res_blocks):
            name_blk = f"{self.name}_res_block_{blk_ix}"
            self.res_blocks.append(
                DRUNetResidualBlock(
                    name=name_blk,
                    filters=self.filters,
                    strides=self.strides,
                    cardinality=self.cardinality,
                    lambda_conv=self.lambda_conv,
                    act=self.act,
                    cbam=self.cbam
                )
                #ResidualBlock(
                #    filters=self.filters,
                #    name=name_blk,
                #    strides=self.strides,
                #    cardinality=self.cardinality,
                #    act=self.act,
                #    channel_expansion_factor=self.channel_expansion_factor,
                #    lambda_conv=self.lambda_conv,
                #)
            )
                
    def call(self, inputs, training=None):
        x = inputs
        for blk_ix in range(self.n_res_blocks):
            x = self.res_blocks[blk_ix](x, training)
        return x
                
                
class DecoderBlock(layer_cls, tm.Logged):
    def __init__(
        self,
        filters:int,
        strides:int=1,
        upsampling_size:Union[Tuple[int], int]=(2,2),
        n_res_blocks:int=2,
        cardinality:int=8,
        lambda_conv:bool=False,
        act:str='relu',
        cbam:bool=False,
        #channel_expansion_factor:int=2,
        *args, **kwargs
    ):
        """ 
        Decoder block for upsampling and residual convolutions.
        
        Accepts inputs both from the previous decoder block and from the spatially
        larger feature maps originating in the encoder blocks through skip connections.
        Upsamples the decoder inputs to the same size as the encoder skip feature maps,
        concatenates these and generates new feature maps with residual convolutions.

        Parameters
        ----------
        filters :  
            The number of convolutional filters to be used in the residual blocks.
            these will output 2 * filters.
        strides : 
            The stride length for the residual conv block
        upsampling_size : optional
            The h, w factors used to upsample the input decoder feature maps
        n_res_blocks : optional
            The number of residual conv blocks to be used
        cardinality : optional
            The cardinality (number of groups) of the grouped convolutions in the Residual
            blocks (ResNeXt-style). ignored if lambda_conv == True.
        lambda_conv : optional
            Boolean flag to enable experimental lambda convolutions instead of 3x3 (grouped) 
            convolutions within the residual blocks.
        act : optional
            String activation function name
        cbam:
            Enables convolutional block attention module in the residual blocks
        args : 
            passed to Layer constructor
        kwargs : 
            passed to Layer constructor
        """
        super(DecoderBlock, self).__init__(*args, **kwargs)
        self.filters = filters
        self.strides = strides
        self.n_res_blocks = n_res_blocks
        self.cardinality = cardinality
        self.lambda_conv = lambda_conv
        self.act = act
        self.cbam = cbam

    def build(self, input_shape):
        # upsample prev decoder ouput and concatenate with encoder output skip
        usc_name = f"{self.name}_upsample_concat"
        self.upsample_concat = UpsampleConcatBlock(
            name=usc_name,
        )
        self.res_blocks = []
        for blk_ix in range(self.n_res_blocks):
            name_blk = f"{self.name}_res_block_{blk_ix}"
            self.res_blocks.append(
                DRUNetResidualBlock(
                    name=name_blk,
                    filters=self.filters,
                    strides=self.strides,
                    cardinality=self.cardinality,
                    lambda_conv=self.lambda_conv,
                    act=self.act,
                    cbam=self.cbam
                )
                #ResidualBlock(
                #    filters=self.filters,
                #    name=name_blk,
                #    strides=self.strides,
                #    cardinality=self.cardinality,
                #    act=self.act,
                #    channel_expansion_factor=self.channel_expansion_factor,
                #    lambda_conv=self.lambda_conv,
                #)
            )
        
    def call(self, inputs, training=None):
        x, xskip = inputs
        # upsample the previous decoder output and concatenate with encoder skip
        x = self.upsample_concat([x, xskip])
        # pass through residual blocks
        for blk_ix in range(self.n_res_blocks):
            x = self.res_blocks[blk_ix](x, training)
        return x


class ExpandAs(layer_cls):
    def __init__(self, name, n_repeats, axis, **kwargs):
        super(ExpandAs, self).__init__(name=name, **kwargs)
        self.n_repeats = n_repeats
        self.axis = axis
        
    def build(self, input_shape):
        self.expander = layers.Lambda(
            lambda x, reps:
                K.repeat_elements(x, reps, axis=self.axis),
                arguments={'reps':self.n_repeats},
                name=self.name + '_Lambda'
        )
    def call(self, inputs):
        return self.expander(inputs)
    
    
def expand_as(tensor, rep, ax, name):
    repeated_tensor = tf.keras.layers.Lambda(
        lambda x, repnum: 
            K.repeat_elements(x, repnum, axis=ax),
            arguments={'repnum': rep},
            name=name
    )(tensor)
    return repeated_tensor        


class GridAttentionBlock(layer_cls, tm.Logged):
    """
    tf.keras Model implementation of an additive Attention Gating Block.
    
    See the papers:
    https://arxiv.org/pdf/1804.03999.pdf
    https://arxiv.org/pdf/1810.07842.pdf
    """
    def __init__(self,
                 name:str,
                 inter_channels:Optional[int]=None,
                 sub_sample_factor=(2,2),
                 *args, **kwargs):
        """
        Parameters:
            in_channels: int
                The number of filters in each encoder block. The outputs of an encoder block
                serve as the input signal (x) of the attention block.
            gating_channels: int
                The number of filters from a decoding block before the corresponding block of the in_channels.
                The outputs of this block serve as the gating signal (g) of the attention block.
            inter_channels: int 
                The number of filters that are generated from x and g to enable adding the output filters. If no
                input is provided, inter_channels is considered in_channels/2. 
            sub_sample_factor: tuple of int
                The rate of down-sampling x before adding to g
        """
        super(GridAttentionBlock, self).__init__(name=name, *args, **kwargs) 
        self.sub_sample_factor = sub_sample_factor
        self.sub_sample_kernel_size = self.sub_sample_factor
        self.inter_channels = inter_channels

    def build(self, input_shape):

        # x corresponds to the input skip line (higher resolution, fewer channels)
        # g corresponds to the input from the decoder (lower resolution, more channels)
        self.x_shape, self.g_shape = input_shape
        #print("x shape:", self.x_shape, "g shape:", self.g_shape)
        # set dimension of intermediate key/query vectors to default to the same as the 
        # gating channels if not set
        if self.inter_channels is None:
            self.inter_channels = self.g_shape[-1]
        # TODO: generalise this to multi-head?
        # Theta^T * x_ij + Phi^T * gating_signal + bias  ==  K + Q + bias
        # These are the weights that generate keys from pixels of x_ij; K := Theta^T * x_ij
        # the dimensions of the keys are then inter_channels. 
        # the kernel size is (2,2) here so that theta * X is downsampled to the same size
        # as the gating signal from the decoder (the queries)
        # is it also worth generalising this since it assumes that x is spatially twice the size of g
        # could do a billinear etc flexible feature map resampling then a conv to get inter_channels
        self.theta = layers.Conv2D(
            name=f"{self.name}_theta",
            filters=self.inter_channels,
            kernel_size=self.sub_sample_kernel_size, 
            padding="valid",
            strides=self.sub_sample_factor #
        )
        # These are the weights that generate queries from pixels of g; Q := Phi^T * g_ij
        # the dimensions of the queries are then inter_channels
        self.phi = layers.Conv2D(
            name=f"{self.name}_phi",
            filters=self.inter_channels,
            kernel_size=1,
            padding="valid",
            strides=1
        )
        # Conv with one filter to generate similarity scores to go into sigmoid
        self.psi = layers.Conv2D(
            name=f"{self.name}_psi",
            filters=1,
            kernel_size=1,
            padding="valid",
            strides=1
        )
        # is this final output conv necessary? the output channels are already the correct shape
        self.attgateconv = layers.Conv2D(
            name=f"{self.name}_attgate",
            filters=self.x_shape[-1],#self.in_channels,
            kernel_size=1,
            padding="valid",
            strides=1
        )
        
        self.add = layers.Add()
        self.multiply = layers.Multiply()
        self.expand_as = ExpandAs(name=f"{self.name}_rep_tnsr", n_repeats=self.x_shape[3], axis=-1)
        self.relu = layers.Activation('relu', name=f"{self.name}_relu")
        self.sigmoid = layers.Activation('sigmoid', name=f"{self.name}_sigmoid")
        self.batch_norm = layers.BatchNormalization(name=f"{self.name}_bn")        
    
    def call(self, inputs, training=None):        
        x, g = inputs
        
        # generate keys (downsample encoder outputs x to match g spatially and producing
        # inter_channel dimensional key maps)
        k = self.theta(x)

        #  Relu(theta_x + phi_g + bias) -> f 
        # generate queries (generate inter_channel dimensional query features from the
        # gating signal [coarse decoder block])
        q = self.phi(g)
        #phi_g = tf.keras.layers.UpSampling2D(size=(1), interpolation='bilinear')(q)
        f = self.relu(self.add([k, q]))
        # collapse intra_channels dimensions of relu(K+Q) down to one channel dimension
        similarity_scores = self.psi(f)
        
        # transform these to attention coefficients in [0,1] at each location with sigmoid
        attention_map = self.sigmoid(similarity_scores)
        
        # upsample the attentions to the original resolution of x
        upsampled_attention_map = layers.UpSampling2D(
            size=(self.x_shape[1]//self.g_shape[1], self.x_shape[2]//self.g_shape[2]),
            interpolation='bilinear'
        )(attention_map)
    
        # repeat the attention map coefficients per channel?
        # just broadcast this with einops?
        channel_expanded_attn_map = self.expand_as(upsampled_attention_map)#, self.x_shape[3], -1, name=f"{self.name}_rep_tnsr")
        # output is the attn coefficents * the inputs x
        y = self.multiply([channel_expanded_attn_map, x])
        # why are these convolutions necessary? -> Mahdi: the 'attgateconv' has been called output transform an serves as an extra  
        #                                          layer of feature mapping but it can be ignored as the output shape is already correct      
        y = self.attgateconv(y)
        return self.batch_norm(y, training=training)

    
class Bottleneck(layer_cls):
    def __init__(
        self,
        name:str,
        #n_ch_in:int,
        n_ch_mid:int,
        *args, **kwargs):
        """ 
        Parameters
        ----------
        name: str
            string name of layer in model
        n_ch_in: int 
            number of channels of a given set of feature maps which is also the
            number of neurons of the output Dense layer
        n_ch_mid: int
            number of channels (i.e. neurons) of the middle Dense layer of the
            Bottleneck block
        """
        super(Bottleneck, self).__init__(name=name, *args, **kwargs)
        #self.n_ch_in = n_ch_in
        self.n_ch_mid = n_ch_mid

    def build(self, input_shape):
        n_ch_in = input_shape[-1]
        # ---- fully-connected network for weighting the channels 
        # the network consists of two fully-connected (i.e. dense) layers for   
        # learning a weight vector for the channels of a given set of feature maps.
        self.mid_layer = layers.Dense(
            self.n_ch_mid, activation='relu', name=f"{self.name}_mid_layer", bias_initializer='zeros')
        self.out_layer = layers.Dense(
            n_ch_in, activation='linear', name=f"{self.name}_out_layer", bias_initializer='zeros')        
        
    def call(self, inputs):
        #print(K.int_shape(inputs))
        r = self.out_layer(self.mid_layer(inputs))    
        #print(K.int_shape(r))
        return r
    
    
class ChannelAttention(layer_cls):
    def __init__(
        self,
        name:str,
        reduction_ratio:int=8,
        *args, **kwargs
    ):
        """ 
        Parameters
        ----------
        name: str
            string name of layer in model
        reduction_ratio: int
            ration of the input channesl to the middle channels in the Bottleneck block
        """
        super(ChannelAttention, self).__init__(name=name, *args, **kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        n_ch_in = input_shape[-1]
        self.middle_layer_size = int(n_ch_in / float(self.reduction_ratio))
        self.bottleneck = Bottleneck(
            name=self.name,
            n_ch_mid=self.middle_layer_size
        )
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.max_pool = layers.GlobalMaxPool2D()
        self.add = layers.Add()
        self.sigmoid = layers.Activation('sigmoid')
        self.reshape = layers.Reshape((1, 1, n_ch_in))
        self.expand_1 = ExpandAs(name=f"{self.name}_rep_dim1", n_repeats=input_shape[1], axis=1)
        self.expand_2 = ExpandAs(name=f"{self.name}_rep_dim2", n_repeats=input_shape[2], axis=2)
        
    def call(self, inputs):
        #print(f'in_shape={in_shape}')
        # Compute the global average- and max-pooling versions of a given set 
        # of feature maps which will be fed into the Bottleneck block  
        avg_pool = self.avg_pool(inputs)
        #print(f'avg_shape={avg_pool.shape}')
        max_pool = self.max_pool(inputs)

        avg_pool_btlnk = self.bottleneck(avg_pool)
        #print(f'avg_btlnk_shape={avg_pool_btlnk.shape}')
        max_pool_btlnk = self.bottleneck(max_pool)

        pool_sum = self.add([avg_pool_btlnk, max_pool_btlnk])
        # TODO: think about adding bias (minor point)
        sig_pool = self.sigmoid(pool_sum)
        sig_pool = self.reshape(sig_pool)
        #print(f'sig_pool_shape={sig_pool.shape}')
        # The computed channel weights should be repeated using the 'expand_as' function
        # to have a tensor of the same shape as the input tensor  
        # TODO: broadcast this to save VRAM
        out1 = self.expand_1(sig_pool)
        #print(f'out1_shape={out1.shape}')
        out2 = self.expand_2(out1)
        #print(f'out2_shape={out2.shape}')

        return out2    
    
    
class SpatialAttention(layer_cls):
    """
    The spatial attention module described in https://arxiv.org/pdf/1807.06521.pdf

    Obtains a simplified aggregate descriptor of the input feature maps using 
    max pooling and average pooling, and creates a spatial attention map by learning
    a large kernel-size convolution with sigmoid activation which is applied to 
    these descriptors to produce a one-channel attention map.
    """
    def __init__(self, name:str, kernel_size=(7,7), *args, **kwargs):
        self.kernel_size = kernel_size
        super(SpatialAttention, self).__init__(name=name, *args, **kwargs)

    def build(self, input_shape):
        # calculate average and max values across the channel dims
        self.ch_avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.ch_max_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))
        self.concat = layers.Concatenate(axis=-1)
        # sigmoid conv maps aggregated channel features to spatial attention coefficients
        self.conv_sig = layers.Conv2D(
            filters=1,
            kernel_size=self.kernel_size,
            padding="same",
            activation="sigmoid",
            strides=(1,1)
        )
        
    def call(self, inputs):
        # aggregate channel feature activations and concatenate to compact descriptor
        chn_avg = self.ch_avg_pool(inputs)
        chn_max = self.ch_max_pool(inputs)
        chn_descriptor = self.concat([chn_avg, chn_max])
        # produce [0, 1] attention coefficients per pixel
        spatial_attn_map = self.conv_sig(chn_descriptor)
        return spatial_attn_map
    
    
class MergeInputGating(layer_cls):
    """
    Merges two sets of feature maps (one larger from encoder, one smaller from decoder)
    by upsampling the smaller then projecting them into a common channel space.
    In this block, the spatial dimension of g is first increased by a factor of 2
    in order to have the same number of pixels in each feature map as x (encoder
    feature maps). 
    The nubmber of channels of g then becomes self.inter_ch using 1D convs.
    At this point, x and g can be added and the obtained set of feature maps will be 
    used for channel-spatial attention.
    """
    def __init__(self, 
                 name:str,
                 inter_ch:int,
                 upsample_factor=(2,2),
                 act:Optional[str]='relu',
                 *args, **kwargs):
        super(MergeInputGating, self).__init__(name=name, *args, **kwargs)
        self.inter_ch = inter_ch
        self.act = act
        self.upsample_factor = upsample_factor

    def build(self, input_shape):
        # x corresponds to the input skip line (higher resolution, fewer channels)
        # g corresponds to the input from the decoder (lower resolution, more channels)
        self.x_shape, self.g_shape = input_shape
        #print("MergeInputGating input shape x:", self.x_shape, " g:", self.g_shape)
        #print("upsample factor is: ", self.upsample_factor)
        self.upsample_g = layers.UpSampling2D(size=self.upsample_factor, interpolation='bilinear')
        # TODO: investigate adding extra conv to x to derive "keys"
        self.phi = layers.Conv2D(
            name=f"{self.name}_phi",
            filters=self.inter_ch,
            kernel_size=1,
            padding="valid",
            strides=(1,1)
        )
        self.theta = layers.Conv2D(
            name=f"{self.name}_theta",
            filters=self.inter_ch,
            kernel_size=1, 
            strides=(1,1)
        )
        self.add = layers.Add()
        if self.act:
            self.act_layer = layers.Activation(self.act)
        
    def call(self, inputs):
        x, g = inputs
        g = self.upsample_g(g)
        #print("MergeGate: upsampled g shape:", g.shape)
        phi_g = self.phi(g)
        theta_x = self.theta(x)
        merged_filters = self.add([theta_x, phi_g])
        if self.act:
            return self.act_layer(merged_filters)
        return merged_filters
    
    
class CSAG(layer_cls):
    """
    Channel-Spatial Attention Gate for segmentation decoder.

    Enhances salient regions and channels of encoder feature maps by 
    reweighting these sequentially with channel, then spatial attention
    coefficients. These coefficients are learned by assimilating information 
    from the semantically rich, spatially coarse decoder feature maps (g) with
    the less abstract, spatially fine feature maps (x) from the encoder skip lines.

    The logic follows that of the Convolutional Block Attention Module (CBAM), 
    described in https://arxiv.org/pdf/1807.06521.pdf, adapted to two distinct 
    input feature maps (x, g) by first combining these in a merging gate.
    """
    def __init__(self,
                 name:str,
                 inter_ch=None,
                 reduction_ratio:Optional[int]=8,
                 merge_act:Optional[str]='relu',
                 spatial_attn_kernel_size:Tuple[int]=(7,7),
                 upsample_factor:tuple=(2,2),
                 *args, **kwargs):
        """
        Parameters:
            name: str
                string name of the CSAG layer in model
            inter_ch: int 
                The number of intermediate channels for merging the input (x) and gating (g) channels. 
                If no input is provided, inter_channels is considered to be in_channels.
            reduction_ratio: int
                The ratio by which to reduce the number of channels in the channel attention 
                bottleneck MLP 
            merge_act: str
                The activation function to be used in the merging gate if any
            spatial_attn_kernel_size:
                The kernel size used in the spatial attention block's sigmoid convolution
            upsample_factor: tuple of int
                The rate of up-sampling g to have the same spatial dimension as x.
        """
        super(CSAG, self).__init__(name=name, *args, **kwargs)
        self.inter_ch = inter_ch
        self.spatial_attn_kernel_size = spatial_attn_kernel_size
        self.reduction_ratio = reduction_ratio
        self.merge_act = merge_act
        self.upsample_factor = upsample_factor

    def build(self, input_shape):
        self.x_shape, self.g_shape = input_shape
        #print("CSAG input shape x:", self.x_shape, " g:", self.g_shape)
        n_ch_in = self.x_shape[-1]
        if self.inter_ch is None:
            self.inter_ch = n_ch_in
        # The input (x) and gating (g) feature maps are first merged (i.e. added) together and the
        # obtained feature maps are then used for the channel and spatial attentions.
        # Having the channel and spatial weights computed, the weighted pixels are multiplied with x.
        self.merge_gate_1 = MergeInputGating(
            self.name+'_merge_1',
            self.inter_ch,
            self.upsample_factor,
            act=self.merge_act
        )
        self.merge_gate_2 = MergeInputGating(
            self.name+'_merge_2',
            self.inter_ch,
            self.upsample_factor,
            act=self.merge_act
        )
        self.channel_attention = ChannelAttention(self.name, reduction_ratio=self.reduction_ratio)
        self.spatial_attention = SpatialAttention(self.name, kernel_size=self.spatial_attn_kernel_size)
        self.multiply_1, self.multiply_2 = layers.Multiply(), layers.Multiply()
        self.bn_1, self.bn_2 = layers.BatchNormalization(), layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        x, g = inputs
        merged_InG = self.merge_gate_1(inputs)
        channel_att = self.channel_attention(merged_InG)
        x_weighted_channels = self.multiply_1([x, channel_att])
        x_weighted_channels = self.bn_1(x_weighted_channels, training=training)
        merge_wchInG = self.merge_gate_2([x_weighted_channels, g])
        spatial_att = self.spatial_attention(merge_wchInG)
        weighted_pixels = self.multiply_2([x_weighted_channels, spatial_att])
        return self.bn_2(weighted_pixels, training=training)


class CBAM(layer_cls):
    """
    Implementation of original Convolutional Block Attention Module (CBAM).
    Described in https://arxiv.org/pdf/1807.06521.pdf.

    This optionally applies as the last step of a residual block in the encoder 
    or decoder, reweighting the residual feature maps.
    """
    def __init__(self, reduction_ratio, kernel_size, *args, **kwargs):
        super(CBAM, self).__init__(*args, **kwargs)
        self.reduction_ratio = reduction_ratio
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.chn_attn_block = ChannelAttention(
            name=self.name + '_ChAttn',
            reduction_ratio=self.reduction_ratio
        )
        self.spt_attn_block = SpatialAttention(
            name=self.name + '_SptAttn',
            kernel_size=self.kernel_size
        )
        self.multiply_1 = layers.Multiply()
        self.multiply_2 = layers.Multiply()

    def call(self, inputs, training=None):
        x = inputs
        # derive channel attention weights from inputs
        ch_attn = self.chn_attn_block(x)
        # reweight inputs by channel attention coefficients
        x_chn_reweighted = self.multiply_1([x, ch_attn])
        # calculate spatial attention weights from reweighted inputs
        sp_attn = self.spt_attn_block(x_chn_reweighted)
        # reweight the channel-reweighted inputs with the spatial attention coefficients
        return self.multiply_2([x_chn_reweighted, sp_attn])
   
