'''
See https://github.com/fchollet/keras/issues/1287

This version has more convolutions and max pooling steps, with larger numbers of kernels

'''

lr = 0.00000001
decay = 0
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
### Convolution layer 1   ###
#############################

model.add(ZeroPadding2D(padding=(60,60),input_shape=(17,136,136)))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(8,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))

#############################
### Convolution layer 2   ###
#############################

model.add(Convolution2D(16,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(16,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))

#############################
### Convolution layer 3   ###
#############################

model.add(Convolution2D(32,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(32,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))

#############################
### Convolution layer 4   ###
#############################

model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='same'))

#############################
##  Deconvolution layer 1  ##
#############################

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(UpSampling2D((2, 2)))

#############################
##  Deconvolution layer 2  ##
#############################

model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(64,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(UpSampling2D((2, 2)))

#############################
##  Deconvolution layer 3  ##
#############################

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(128,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(UpSampling2D((2, 2)))

#############################
##  Deconvolution layer 4  ##
#############################

model.add(Convolution2D(256,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(Convolution2D(256,n_conv,n_conv,border_mode='same',init=init))
model.add(Activation(act))

model.add(UpSampling2D((2, 2)))

#############################
###    Output layer 1     ###
#############################

model.add(Convolution2D(1,1,1,border_mode='same',init=init))
model.add(Activation(act))

model.add(Cropping2D(cropping=((60,60), (60,60))))

#model.summary()