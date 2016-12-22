'''

Another early model
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

lr = 0.0005
decay = 0

model = Sequential()

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th',input_shape=(17,136,136)))
model.add(Convolution2D(100,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(32,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(16,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(8,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(4,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation(act))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(1,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation('relu'))

model.add(ZeroPadding2D(padding=(1,1),dim_ordering ='th'))
model.add(Convolution2D(1,n_conv,n_conv,border_mode='valid',dim_ordering ='th',init=init))
model.add(Activation('relu'))