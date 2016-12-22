'''

Another early deconvolution model
Unclear how this performed
I can't imagine well - I stopped using it

'''

leaky = LeakyReLU()
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
adadelta = Adadelta()
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
act = 'relu'
init = 'he_normal'

lr = 0.00001
decay = 1e-3
n_conv = 3
n_filters = 32
batch_size = 1

model = Sequential()

#############################
### Convolution layers    ###
#############################

model.add(ZeroPadding2D(padding=(1,1),input_shape=(17,136,136)))
model.add(Convolution2D(128,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,32,136,136)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(128,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,32,136,136)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
# (None,32,68,68)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,16,68,68)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,16,68,68)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
# (None,16,34,34)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,8,34,34)

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid'))
# (None,8,17,17)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,17,17)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,17,17)

# Note: batch size must be a divisor of 22 and 6 otherwise validation data can't be used
#batch_size = 16

n_deconv = 1

#############################
### Deconvolution layer 1 ###
#############################
#model.add(ZeroPadding2D(padding=(1,1)))
model.add(Deconvolution2D(16,n_deconv,n_deconv,output_shape=(batch_size,16,34,34),border_mode='valid',init=init,subsample=(2,2)))
model.add(Activation(act))
# (None,4,34,34)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,34,34)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,34,34)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,34,34)

#############################
### Deconvolution layer 2 ###
#############################
#model.add(ZeroPadding2D(padding=(1,1)))
model.add(Deconvolution2D(32,n_deconv,n_deconv,output_shape=(batch_size,32,68,68),border_mode='valid',init=init,subsample=(2,2)))
model.add(Activation(act))
# (None,4,68,68)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,68,68)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,68,68)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,68,68)

#############################
### Deconvolution layer 3 ###
#############################
#model.add(ZeroPadding2D(padding=(1,1)))
model.add(Deconvolution2D(64,n_deconv,n_deconv,output_shape=(batch_size,64,136,136),border_mode='valid',init=init,subsample=(2,2)))
model.add(Activation(act))
# (None,4,136,136)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,136,136)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,136,136)

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(64,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))
# (None,4,136,136)

#############################
### Output layers         ###
#############################

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(2,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1)))
model.add(Convolution2D(1,n_conv,n_conv,border_mode='valid',init=init))
model.add(Activation(act))

model.summary()
