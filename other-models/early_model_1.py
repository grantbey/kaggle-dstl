'''

Early example
Deconvolution makes weird patterns
Abandoned

'''

n_conv = 3

leaky = LeakyReLU()
sgd = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
adagrad = Adagrad(lr=0.001, epsilon=1e-08, decay=0.0)
adadelta = Adadelta()
adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay)
act = 'relu'
init = 'he_normal'

lr = 0.00001
decay = 1e-3


model = Sequential()

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th',input_shape=(17,136,136)))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid',dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid',dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(MaxPooling2D(pool_size=(2, 2),border_mode='valid',dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

# This section flattens and runs Dense() layers
#model.add(Flatten()) # 10*17*17 = 2890 dimensions
#model.add(Dense(2000))
#model.add(Activation(leaky))
#model.add(Dense(1000))
#model.add(Activation(leaky))
#model.add(Dense(2890))
#model.add(Activation(leaky))
#model.add(Reshape((10,17,17)))

# Note: batch size must be a divisor of 22 and 6 otherwise validation data can't be used
#batch_size = 16

#model.add(ZeroPadding2D(padding=(2,2),dim_ordering ='th'))
model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Deconvolution2D(4,3,3,output_shape=(1,4,34,34),border_mode='valid',dim_ordering ='th',init=init,subsample=(2,2)))
model.add(Activation(act))

#model.add(UpSampling2D((2,2),dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Deconvolution2D(8,3,3,output_shape=(1,8,68,68),border_mode='valid',dim_ordering ='th',init=init,subsample=(2,2)))
model.add(Activation(act))

#model.add(UpSampling2D((2,2),dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Deconvolution2D(16,3,3,output_shape=(1,16,136,136),border_mode='valid',dim_ordering ='th',init=init,subsample=(2,2)))
model.add(Activation(act))

#model.add(UpSampling2D((2,2),dim_ordering = 'th'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(1,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))


#model.add(Activation('tanh'))