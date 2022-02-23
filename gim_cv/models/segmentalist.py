from typing import List, Tuple, Union, Optional

import pandas as pd 
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as layers
#keras = tf.keras
#layers = keras.layers

import gim_cv.losses as losses
import gim_cv.models.blocks as blocks

class Segmentalist(keras.Model):
    """
    A flexible encoder-decoder-style segmentation model.
    
    Uses residual blocks and U-Net style skip connections, and is configurable
    to use attention gates in the decoder, multi-scale input pyramid pooling, 
    deep supervision, ResNeXt-style grouped convolutions and Lambda convolutions.
    
    Parameters
    ----------
    n_classes : optional
    
    layer_blocks : optional
    
    last_decoder_layer_blocks : optional
        
    initial_filters : optional
        
    residual_filters : optional    
        
    initial_kernel_size : optional
        
    cardinality : optional
        
    channel_expansion_factor : optional    
        
    act : optional
        
    downsample : optional    
        
    decoder_attention_gates : optional

    encoder_cbam : optional

    decoder_cbam : optional
        
    pyramid_pooling : optional
        
    deep_supervision : optional
        
    lambda_conv : optional
        
    multi_level_skip : optional
    
    name : optional
        
    *args, **kwargs
    """
    def __init__(
        self,
        n_classes:int=1,
        layer_blocks:List[int]=[2,2,2,2],
        last_decoder_layer_blocks:int=2,
        initial_filters:int=64,
        residual_filters:List[int]=[64,128,256,512],
        initial_kernel_size:Tuple[int]=(7,7),
        head_kernel_size:Tuple[int]=(1,1),
        cardinality:int=1,
        #channel_expansion_factor:int=2,
        act:str='relu',
        downsample:str='pool',
        decoder_attention_gates:Optional[str]='SAG',
        encoder_cbam:bool=False,
        decoder_cbam:bool=False,
        pyramid_pooling:bool=False,
        deep_supervision:bool=False,
        lambda_conv:bool=False,
        multi_level_skip:bool=False,
        name='Segmentalist',
        *args, **kwargs
    ):
        super(Segmentalist, self).__init__(name=name, *args, **kwargs)
        self.n_classes = n_classes
        self.layer_blocks = layer_blocks
        self.last_decoder_layer_blocks = last_decoder_layer_blocks
        self.initial_filters = initial_filters
        self.residual_filters = residual_filters
        self.initial_kernel_size = initial_kernel_size
        self.head_kernel_size = head_kernel_size
        self.cardinality = cardinality
        #self.channel_expansion_factor = channel_expansion_factor
        self.act = act
        self.downsample = downsample
        self.valid_attention_gates = ('CSAG', 'SAG', None)
        if decoder_attention_gates in self.valid_attention_gates:
            self.decoder_attention_gates = decoder_attention_gates
        else:
            raise ValueError(
                f"Invalid setting '{decoder_attention_gates}' for attention_gates."
                f" Valid choices are {self.valid_attention_gates}"
            )
        self.encoder_cbam = encoder_cbam
        self.decoder_cbam = decoder_cbam
        self.pyramid_pooling = pyramid_pooling
        self.deep_supervision = deep_supervision
        self.lambda_conv = lambda_conv
        self.multi_level_skip = multi_level_skip
        if self.multi_level_skip:
            raise NotImplementedError("NYI")
        if self.downsample == 'pool':
            self.strides = 1
        elif self.downsample == 'strides':
            self.strides = 2
        else:
            raise ValueError("only 'pool' and 'strides' supported!")
        self.n_blocks = len(self.layer_blocks)
    
    def build(self, input_shape):
        # first large-kernel convolution as in ResNet, usually halves spatial size
        self.initial_conv = blocks.ConvBlock(
            filters=self.initial_filters,
            strides=1,
            kernel_size=self.initial_kernel_size,
            in_shape=input_shape,
            act=self.act,
            name='InitialConv'
        )
        
        # --- build main encoder and decoder blocks --------------------------------------------
        # encoder blocks - residual convolutions and downsampling
        self.encoder_blocks = [
            blocks.EncoderBlock(
                filters=filters,
                n_res_blocks=n_res_blocks,
                cardinality=self.cardinality,
                strides=self.strides,
                cbam=self.encoder_cbam,
                lambda_conv=self.lambda_conv,
                act=self.act,
                #channel_expansion_factor=self.channel_expansion_factor,
                name=f'EncoderBlock_{ix+1}'
            )
            for ix, (filters, n_res_blocks) in enumerate(zip(self.residual_filters, self.layer_blocks))
        ]
        # if using pooling, we have to add a max-pooling stage so that these produce downsampled outputs
        # as they do when using strides, so wrap up each EncoderBlock with a pooling layer
        if self.downsample == 'pool':
            self.encoder_blocks = [
                keras.Sequential(
                    [
                        layers.MaxPool2D(
                            name=f"{self.name}_maxpool_2d_{ix+1}",
                            pool_size=(2,2),
                            strides=2,
                            padding="same"
                        ),
                        e
                    ]
                )
                for ix, e in enumerate(self.encoder_blocks)
            ]        
        # decoder blocks - each performs upsampling of decoder feature maps, concatenation of this with the encoder skip lines,
        # then a block of residual convolutions
        # since the deepest encoder block doubles as the "bridge" block with the highest number of feature maps,
        # the decoder filters are reduced starting from the second largest set of feature maps 
        self.decoder_blocks = [
            blocks.DecoderBlock(
                filters=filters,
                n_res_blocks=n_res_blocks,
                cardinality=self.cardinality,
                strides=1,
                cbam=self.decoder_cbam,
                lambda_conv=self.lambda_conv,
                act=self.act,
                #channel_expansion_factor=self.channel_expansion_factor,
                name=f'DecoderBlock_{ix+1}'
            )
            for ix, (filters, n_res_blocks) in enumerate(zip(
                reversed([self.initial_filters] + self.residual_filters[:-1]), reversed(self.layer_blocks)
            ))
        ]
        # --- optional components
        # --- multi-scale "pyramid" pooling
        # downsample the input at multiple spatial resolutions and provide these as additional inputs to deeper encoder blocks
        # usual implementations don't feed the downsampled inputs at the lowest resolution to the deepest "bridge" block
        if self.pyramid_pooling:
            # average pooling downsamples the image at each layer of the encoder after the first
            self.avg_pool_layers = [
                layers.AveragePooling2D(
                    pool_size=(2, 2),
                    strides=2,
                    name=f'avg_pool_{ix}'
                )
                for ix in range(1, self.n_blocks)
            ]
            # each spatial resolution has its own conv stack
            self.input_pyramid_convs = [
                blocks.ConvBlock(
                    name=f'pyramid_conv_{ix}',
                    filters=self.residual_filters[ix],
                    kernel_size=(3, 3),
                    padding="same",
                    act='relu',
                )
                for ix in range(1, self.n_blocks)
            ]
            self.pyramid_concats = [
                layers.Concatenate(axis=-1)
                for ix in range(1, self.n_blocks)
            ]
        # --- attention gates
        if self.decoder_attention_gates == 'SAG':
            self.attn_blks = [blocks.GridAttentionBlock(name=f"SAG_{ix}") for ix in range(self.n_blocks)]
        elif self.decoder_attention_gates == 'CSAG':
            self.attn_blks = [blocks.CSAG(name=f"CSAG_{ix}") for ix in range(self.n_blocks)]
        # --- deep supervision - convolution to n_classes channels and activation for intermediate decoder outputs
        # to force learning the output mask at multiple scales
        if self.deep_supervision:
            # convolution to produce n_classes feature maps from each decoder output
            self.ds_output_convs = [
                layers.Conv2D(
                    filters=self.n_classes,
                    kernel_size=self.head_kernel_size,
                    padding='same',
                    use_bias=True,
                    kernel_initializer='glorot_uniform',
                    name=f'deep_supervision_output_conv_{ix}',
                )
                #layers.Activation('softmax' if self.n_classes > 1 else 'sigmoid', name=f'deep_supervision_output_act_{ix}')
                for ix in range(1, self.n_blocks)
            ]
            # upsampling to upsample these output feature maps' spatial resolutions
            self.ds_upsampling_layers = [
                layers.UpSampling2D((2,2), interpolation='bilinear', name=f'ds_upsmp_{ix}') for ix in range(1, self.n_blocks)
            ]
            # addition to add upsampled fmaps to those of higher deeply supervised layers
            self.ds_add_layers = [
                layers.Add(name=f'ds_add_{ix}') for ix in range(1, self.n_blocks)
            ]
        # Output conv layer for full-size final decoder feature maps
        # use softmax activation for multiclass, sigmoid for binary
        self.final_output_layer = layers.Conv2D(
                filters=self.n_classes,
                kernel_size=self.head_kernel_size,
                padding='same',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                name='final_conv'
        )
            #keras.Sequential([
            #layers.Activation('softmax' if self.n_classes > 1 else 'sigmoid', name='final_act')
        self.final_act = layers.Activation('softmax' if self.n_classes > 1 else 'sigmoid', name='final_act')

        
    def call(self, inputs, training=None):
        initial_img = inputs
        encoder_outputs, decoder_outputs = [initial_img], []
        encoder_outputs.append(self.initial_conv(initial_img, training=training))
        # if pyramid spatial pooling is enabled, we also need to downsample the input image at each step
        # of the encoder
        if self.pyramid_pooling:
            downsampled_inputs = [initial_img]
        # if deep supervision is enabled, we need the model to return the segmentation masks obtained 
        # from the intermediate decoder outputs
        if self.deep_supervision:
            ds_outputs = []
        # produce encoder feature maps
        # these are calculated sequentially one block at a time starting from the input image batch
        for blk_ix, encoder_block in enumerate(self.encoder_blocks):
            # -- determine the inputs to the encoder blocks
            # if input pyramid spatial pooling is enabled, resample the input image batch at all intermediate 
            # resolutions, generate feature maps from downsampled inputs and concatenate w/ those of prev encoder layer
            if self.pyramid_pooling and 0 < blk_ix < len(self.encoder_blocks):
                downsampled_input = self.avg_pool_layers[blk_ix-1](downsampled_inputs[-1])
                downsampled_inputs.append(downsampled_input)
                downsampled_fmaps = self.input_pyramid_convs[blk_ix-1](downsampled_input, training=training)
                blk_input = self.pyramid_concats[blk_ix-1]([encoder_outputs[-1], downsampled_fmaps])
            # otherwise just pass in the output of the previous encoder layer
            else:
                blk_input = encoder_outputs[-1]
            # -- the main encoder block's residual convolutions
            blk_output = encoder_block(blk_input, training=training)
            encoder_outputs.append(blk_output)
        # the "bridge" - the last set of encoder feature maps become the first set into the decoder
        decoder_outputs = [encoder_outputs[-1]]
        # produce decoder feature maps from the encoder skip connections
        for blk_ix, decoder_block in enumerate(self.decoder_blocks):
            # if attention gates are toggled, these will learn to use the decoder outputs to 
            # dynamically enhance salient regions of the encoder feature maps across the skip lines 
            if self.decoder_attention_gates:
                decoder_skip_input = self.attn_blks[blk_ix](
                    [encoder_outputs[len(encoder_outputs) - 2 - blk_ix], decoder_outputs[-1]],
                    training=training
                )
            # otherwise just the raw skip feature maps are used in the decoder.
            else:
                decoder_skip_input = encoder_outputs[len(encoder_outputs) - 2 - blk_ix]
            blk_output = decoder_block([decoder_outputs[-1], decoder_skip_input], training=training)
            # if deep supervision is enabled, these outputs are trained directly to reproduce the segmentation
            # masks at coarser resolutions, analogously to input pyramid spatial pooling.
            if self.deep_supervision and blk_ix < len(self.decoder_blocks)-1:
                # convolution of decoder outputs down to n_channels outputs 
                ds_out_conv = self.ds_output_convs[blk_ix](blk_output)
                # element-wise addition with previous layer from second decoder layer onwards
                if not ds_outputs:
                    ds_out = ds_out_conv
                else:
                    ds_out = self.ds_add_layers[blk_ix-1]([ds_outputs[-1], ds_out_conv])
                # upsampling to match spatial res of the next layer up
                ds_out = self.ds_upsampling_layers[blk_ix](ds_out)
                # store upsampled features output for next decoder block
                ds_outputs.append(ds_out)
            decoder_outputs.append(blk_output)
        # if deep supervision is on, the final output feature maps from the last decoder block
        # are added to those upsampled from the deeper decoder blocks before final activation
        if self.deep_supervision:
            final_out = self.ds_add_layers[-1]([
                self.final_output_layer(decoder_outputs[-1]),
                ds_outputs[-1]
            ])
        # otherwise just use the final output feature maps from the last decoder block alone to 
        # generate segmentation values
        else:
            final_out = self.final_output_layer(decoder_outputs[-1])
        # softmax/sigmoid activation to obtain class probabilities
        return self.final_act(final_out)
    
    @classmethod
    def load_from_metadata(cls, row : pd.Series, val : bool = True):
        """
        Loads a Segmentalist with the weights loaded from the checkpoint with 
        the lowest validation loss.

        If val is False, uses the checkpoint with the lowest training loss.
        
        Parameters
        ----------
        row :
            Row of a pandas dataframe generated from a saved model directory
            using the utility function (ATM bin/)utils.collate_run_data.
            This row contains hyperparameters and training setup associated
            with a given set of weights.
        val : Boolean flag to select whether to load the weights associated 
            with the lowest validation or training loss checkpoints
        """
        if row.sag:
            decoder_ag = 'SAG'
        elif row.csag:
            decoder_ag = 'CSAG'
        else:
            decoder_ag = None
        model = Segmentalist(
            n_classes=1,
            layer_blocks=row.layer_blocks_,
            last_decoder_layer_blocks=row.last_decoder_layer_blocks,
            initial_filters=row.initial_filters,
            residual_filters=row.residual_filters_,
            initial_kernel_size=row.initial_kernel_size_,
            head_kernel_size=row.head_kernel_size_,
            cardinality=row.cardinality,
            #channel_expansion_factor:int=2,
            act='relu',
            downsample='pool',
            decoder_attention_gates=decoder_ag,
            encoder_cbam=row.encoder_cbam,
            decoder_cbam=row.decoder_cbam,
            pyramid_pooling=row.pyramid_pooling,
            deep_supervision=row.deep_supervision,
            lambda_conv=row.lambda_conv,
            multi_level_skip=False
        )
        model.checkpoint_uuid = row.uuid4
        # use same setup as training script to ensure parity in loaded model
        # interpret optimizer
        if row.optimiser == 'sgd':
            opt = tf.keras.optimizers.SGD(
                learning_rate=row.lr_init, momentum=0.85, nesterov=False
            )
        elif row.optimiser == 'adam':
            opt = tf.keras.optimizers.Adam(
                learning_rate=row.lr_init, beta_1=0.9, beta_2=0.999, amsgrad=False
            ) # check out RADAM?
        elif row.optimiser == 'ranger':
            radam = tfa.optimizers.RectifiedAdam(lr=row.lr_init, min_lr=row.lr_min)
            opt = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
        else:
            raise ValueError(f"Optimiser {opt} not understood")
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
    
    