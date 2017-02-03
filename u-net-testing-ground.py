
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

class_ = sys.argv[1]
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
run = set_counter(sys.argv[2])
print('This is run # %i' %run)

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
push('Training is done on class %i' %class_,
     'Train loss: %f, train jaccard: %f, val loss %f, val jaccard%f' %(model.history.history['loss'][-1],model.history.history['jaccard'][-1],model.history.history['val_loss'][-1],model.history.history['val_jaccard'][-1]))
