
# coding: utf-8

# ##### U-Net architecture
# 
# See [here](https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py#L19) for code and [here](https://arxiv.org/pdf/1505.04597.pdf) for the original literature.

import numpy as np
from keras.models import Model
from keras.layers import Input, merge, Activation, Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D, BatchNormalization, Dropout
from keras.optimizers import Adadelta, Adam
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras import backend as K
from keras.regularizers import l2
from keras.constraints import maxnorm
from keras.layers.advanced_activations import LeakyReLU

K.set_image_dim_ordering('th')  # Theano dimension ordering in this code
# "tf" assumes (rows, cols, channels) while "th" assumes (channels, rows, cols)
# Possibly change this around natively in the data so the backend doesn't have to switch them
# Only necessary if I use TF!

from matplotlib import pyplot as plt
from pushbullet import Pushbullet

import sys

# ### Helper functions

# Pushbullet notifier
def push(title='Done!',text=''):
    Pushbullet('o.YFPNNPfGRekivaCGHa4qMSgjZt8zJ6FL').devices[0].push_note(title,text)
    
# Import the training data
def import_data(class_):
    x = np.load('./data/x_augmented.npy','r')
    y = np.load('./data/y_augmented.npy','r')
    y_oneclass = y[:,class_:class_+1,...]
    '''
    Classes:
    0 Buildings - large building, residential, non-residential, fuel storage facility, fortified building
    1 Misc. Manmade structures 
    2 Road 
    3 Track - poor/dirt/cart track, footpath/trail
    4 Trees - woodland, hedgerows, groups of trees, standalone trees
    5 Crops - contour ploughing/cropland, grain (wheat) crops, row (potatoes, turnips) crops
    6 Waterway 
    7 Standing water
    8 Vehicle Large - large vehicle (e.g. lorry, truck,bus), logistics vehicle
    9 Vehicle Small - small vehicle (car, van), motorbike
    '''    
    return x, y, y_oneclass

class_ = int(sys.argv[1])
x, y, y_oneclass = import_data(class_)

# Increment the counter
def counter():
    run = np.load('./data/run_counter.npy')
    run += 1
    np.save('./data/run_counter.npy',run)
    return run
run = counter()

# Set the counter to a specific value
def set_counter(run):
    run = run
    np.save('./data/run_counter.npy',run)
    return run
# Uncomment the next line to manually set the counter if something goes wrong
run = set_counter(int(sys.argv[2]))
print('This is run # %i' %run)

def compiler(img_rows = x.shape[2],img_cols = x.shape[3],
            nfilters = 32,activation = 'relu',init = 'he_normal',
            lr=1.0,decay=0.0,momentum=0.0, nesterov=False,reg=0.01,p=[0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2]):
    
    def jaccard(y_true, y_pred,smooth=1.):
        y_true_f = K.flatten(y_true)
        y_pred_f = K.flatten(y_pred)
        intersection = K.sum(y_true_f * y_pred_f)
        return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)
    
    def Conv2DReluBatchNorm(n_filter, w_filter, h_filter, inputs, activation, init='he_uniform',dropout=0.2):
        # Batch norm after activation / leakyrelu
        #return BatchNormalization(mode=2, axis=1)(LeakyReLU()((Convolution2D(n_filter, w_filter, h_filter, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(inputs))))
        
        # Batch norm before activation
        #return LeakyReLU()(BatchNormalization(mode=0, axis=1)((Convolution2D(n_filter, w_filter, h_filter, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(inputs))))
        
        # Batch norm after activation / relu
        return BatchNormalization(mode=2, axis=1)(Activation(activation=activation)((Convolution2D(n_filter, w_filter, h_filter, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(inputs))))
        
    def up_conv(nfilters,filter_factor,inputs,init=init,activation=activation):
        # No batch norm
        #return LeakyReLU()(Convolution2D(nfilters*filter_factor, 2, 2, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(UpSampling2D(size=(2, 2))(inputs)))
        
        # Batch norm after activation
        #return BatchNormalization(mode=2, axis=1)(LeakyReLU()(Convolution2D(nfilters*filter_factor, 2, 2, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(UpSampling2D(size=(2, 2))(inputs))))
        
        # Batch norm after activation / relu
        return BatchNormalization(mode=2, axis=1)(Activation(activation=activation)(Convolution2D(nfilters*filter_factor, 2, 2, border_mode='same',init=init,W_regularizer=l2(reg),W_constraint = maxnorm(3))(UpSampling2D(size=(2, 2))(inputs))))

    inputs = Input((20, img_rows, img_cols))
    padded = ZeroPadding2D(padding=(12,12))(inputs)
    
    conv1 = Conv2DReluBatchNorm(nfilters, 3, 3, padded, activation=activation,init=init)
    conv1 = Conv2DReluBatchNorm(nfilters, 3, 3, conv1, activation=activation,init=init)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(p=p[0])(pool1)

    conv2 = Conv2DReluBatchNorm(nfilters*2, 3, 3, pool1, activation=activation,init=init)
    conv2 = Conv2DReluBatchNorm(nfilters*2, 3, 3, conv2, activation=activation,init=init)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(p=p[1])(pool2)

    conv3 = Conv2DReluBatchNorm(nfilters*4, 3, 3, pool2, activation=activation,init=init)
    conv3 = Conv2DReluBatchNorm(nfilters*4, 3, 3, conv3, activation=activation,init=init)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(p=p[2])(pool3)

    conv4 = Conv2DReluBatchNorm(nfilters*8, 3, 3, pool3, activation=activation,init=init)
    conv4 = Conv2DReluBatchNorm(nfilters*8, 3, 3, conv4, activation=activation,init=init)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(p=p[3])(pool4)

    conv5 = Conv2DReluBatchNorm(nfilters*16, 3, 3, pool4, activation=activation,init=init)
    conv5 = Conv2DReluBatchNorm(nfilters*16, 3, 3, conv5, activation=activation,init=init)
    conv5 = Dropout(p=p[4])(conv5)
        
    up6 = merge([up_conv(nfilters,8,conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2DReluBatchNorm(nfilters*8, 3, 3, up6, activation=activation,init=init)
    conv6 = Conv2DReluBatchNorm(nfilters*8, 3, 3, conv6, activation=activation,init=init)
    conv6 = Dropout(p=p[5])(conv6)

    up7 = merge([up_conv(nfilters,4,conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2DReluBatchNorm(nfilters*4, 3, 3, up7, activation=activation,init=init)
    conv7 = Conv2DReluBatchNorm(nfilters*4, 3, 3, conv7, activation=activation,init=init)
    conv7 = Dropout(p=p[6])(conv7)

    up8 = merge([up_conv(nfilters,2,conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2DReluBatchNorm(nfilters*2, 3, 3, up8, activation=activation,init=init)
    conv8 = Conv2DReluBatchNorm(nfilters*2, 3, 3, conv8, activation=activation,init=init)
    conv8 = Dropout(p=p[7])(conv8)

    up9 = merge([up_conv(nfilters,1,conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2DReluBatchNorm(nfilters, 3, 3, up9, activation=activation,init=init)
    conv9 = Conv2DReluBatchNorm(nfilters, 3, 3, conv9, activation=activation,init=init)
    conv9 = Dropout(p=p[8])(conv9)
    
    conv10 = Conv2DReluBatchNorm(1, 1, 1, conv9, activation='relu',init=init)
    cropped = Cropping2D(cropping=((12,12), (12,12)))(conv10)
    output = Activation(activation='sigmoid')(cropped)
    
    model = Model(input=inputs, output=output)
    
    model.compile(optimizer=Adam(lr=lr,decay=decay), loss='binary_crossentropy', metrics=[jaccard])
    
    return model

p=[0.1,0.2,0.3,0.4,0.5,0.4,0.3,0.2,0.1] # current version
#p=[0.2,0.3,0.4,0.5,0.5,0.5,0.4,0.3,0.2] # symmetric but more dropout
#p=[0.2,0.2,0.3,0.3,0.4,0.4,0.5,0.5,0.6] # increasing

model = compiler(img_rows=x.shape[2],img_cols=x.shape[3],
            nfilters=16,activation='relu',init='he_normal',
            lr=0.001,decay=0,momentum=0,reg=0,p=p)

def trainer(model,fit=True,use_existing=False):
    print('This is run # %i' %run)
    
    if use_existing:
        model.load_weights('./data/model_weights_class_{}_run_{}.hdf5'.format(_class,run))
        
    if fit:
        quitter = EarlyStopping(monitor='loss', min_delta=0.001, patience=100, verbose=1, mode='auto')
        lrreducer = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, verbose=1, mode='auto', epsilon=0.001, cooldown=2, min_lr=0)
        model_checkpoint = ModelCheckpoint('./data/model_weights_class_{}_run_{}.hdf5'.format(class_,run), monitor='loss', save_best_only=True)
        csvlogger = CSVLogger('./data/training_log_run_'+str(run), separator=',', append=True)

        model.fit(x, y_oneclass,
                  batch_size=20,
                  nb_epoch=1000,
                  verbose=1,
                  shuffle=True,
                  callbacks=[model_checkpoint,csvlogger],
                  validation_split=0.2,
                  initial_epoch=0)
            
    preds = model.predict(x, verbose=1)
    np.save('preds.npy', preds)
    
    return model

model = trainer(model,fit=True,use_existing=False)
model.save('u-net-complete-model-run_{}_class_{}.h5'.format(run,class_))
push('Training is done',
     'Train loss: %f, train jaccard: %f, val loss %f, val jaccard%f' %(model.history.history['loss'][-1],model.history.history['jaccard'][-1],model.history.history['val_loss'][-1],model.history.history['val_jaccard'][-1]))
