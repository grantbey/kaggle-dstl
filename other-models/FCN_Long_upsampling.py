'''

Based on FCN by LONG et al.
But uses upsampling instead of Deconvolution2D
See: https://github.com/shelhamer/fcn.berkeleyvision.org/blob/master/voc-fcn32s/net.py

Slow and patchy patterns again

'''
lr = 0.000001
decay = 1e-2
n_conv = 3
n_filters = 32
batch_size = 1

leaky = LeakyReLU()
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
adadelta = Adadelta()
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
act = 'relu'
init = 'he_normal'

model = Sequential()

#############################
### Convolution layers    ###
#############################

model.add(ZeroPadding2D(padding=(60,60),input_shape=(17,136,136)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,32,136,136)

model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,32,136,136)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
# (None,32,68,68)

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,16,68,68)

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,16,68,68)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
# (None,16,34,34)

model.add(Convolution2D(256,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(Convolution2D(256,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(Convolution2D(256,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
# (None,8,17,17)

model.add(Convolution2D(512,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(Convolution2D(512,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(Convolution2D(512,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))
# (None,512,16,16) this is updated :)

###############################

model.add(Convolution2D(4096,7,7,border_mode='same',init=init))
model.add(Activation(act))

model.add(Dropout(0.5))

model.add(Convolution2D(4096,1,1,border_mode='same',init=init))
model.add(Activation(act))

model.add(Dropout(0.5))

model.add(Convolution2D(1,1,1,border_mode='same',init=init))
model.add(Activation(act))

model.add(UpSampling2D((16,16)))
#model.add(Deconvolution2D(1,16,16,output_shape=(batch_size,1,256,256),border_mode='same',init=init,subsample=(16,16),bias=False))
#model.add(Activation(act))

model.add(Cropping2D(cropping=((60,60), (60,60))))

#model.summary()