#import
#===================
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout,Input, AveragePooling2D,BatchNormalization,Activation,ZeroPadding2D,concatenate,Add
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
#===================
def cnn(waferSizeX, waferSizeY):
#===================
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same',input_shape=(waferSizeX, waferSizeY, 3)))  					# 卷積層 (輸入)
    model.add(Dropout(0.25))           							    # Dropout
    model.add(MaxPooling2D((2, 2)))                        			# 池化層
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))  # 卷積層
    model.add(Dropout(0.25))                              			# Dropout 層
    model.add(MaxPooling2D((2, 2)))                             		# 池化層
    model.add(Flatten())                                    			# 展平層
    model.add(Dropout(0.25))                                   		# Dropout
    model.add(Dense(1024, activation='relu'))                      	# 密集層
    model.add(Dropout(0.25))                                     	    # Dropout
    model.add(Dense(8, activation='softmax'))                    		# 密集層 (輸出分類)
    return model
#===================
def LeNet(waferSizeX, waferSizeY):
#===================
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(waferSizeX, waferSizeY, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(8, activation='softmax'))
    return model
#===================
def AlexNet(waferSizeX, waferSizeY):
#===================
    model = Sequential()
    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(waferSizeX,waferSizeY,3), kernel_size=(11,11),strides=(4,4), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
    model.add(BatchNormalization()) ## Batch Normalisation before passing it to the next layer

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11,11), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')) # Pooling
    model.add(BatchNormalization()) # Batch Normalisation

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization())# Batch Normalisation

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(BatchNormalization()) # Batch Normalisation

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='valid'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid')) # Pooling
    model.add(BatchNormalization()) # Batch Normalisation

    # Passing it to a dense layer
    model.add(Flatten())
    # 1st Dense Layer
    model.add(Dense(4096, input_shape=(224*224*3,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # Add Dropout to prevent overfitting
    model.add(BatchNormalization()) # Batch Normalisation

    # 2nd Dense Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # Add Dropout
    model.add(BatchNormalization()) # Batch Normalisation

    # 3rd Dense Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    model.add(Dropout(0.4)) # Add Dropout
    model.add(BatchNormalization()) # Batch Normalisation

    # Output Layer
    model.add(Dense(8)) # was 17
    model.add(Activation('softmax'))
    return model
    
#===================
def vgg16(waferSizeX, waferSizeY):
#===================
    vgg16_model = VGG16(include_top=False,weights='imagenet',input_shape=(waferSizeX,waferSizeY,3))
    model = Sequential()
    model.add(vgg16_model)    # 將 vgg16 做為一層
    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(8, activation='softmax'))
    vgg16_model.trainable = False     # 凍結權重
    return model

#===================
#Resnet18
#===================
def basic_block(x,filters,stride,name):
	shortcut = x
	x = Conv2D(filters,(3,3),strides=stride,padding='same')(x)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(filters, (3, 3), strides=1,padding='same' )(x)
	x = BatchNormalization(axis=3)(x)
	if x.shape != shortcut.shape:
		shortcut = Conv2D(filters,(1,1),strides = stride,name=name)(shortcut)
		shortcut = BatchNormalization(axis=3)(shortcut)
	x = Add()([x,shortcut])
	x = Activation('relu')(x)
	return x

def Resnet18(waferSizeX, waferSizeY):
    input = Input((waferSizeX, waferSizeY, 3))
    x = ZeroPadding2D((3,3))(input)
    x = Conv2D(64,(7,7),strides=2)(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(3,3),strides = (2,2),padding='same')(x)
    x = basic_block(x,64,1,name='shortcut1')
    x = basic_block(x,64,1,name='shortcut2')
    x = basic_block(x, 128, 2,name='shortcut3')
    x = basic_block(x, 128, 1,name='shortcut4')
    x = basic_block(x, 256, 2,name='shortcut5')
    x = basic_block(x, 256, 1,name='shortcut6')
    x = basic_block(x, 512, 2,name='shortcut7')
    x = basic_block(x, 512, 1,name='shortcut8')
    size = int(x.shape[1])
    x = AveragePooling2D(pool_size=(size,size))(x)
    x = Flatten()(x)
    x = Dense(8,activation='softmax')(x)
    model = Model(inputs = input,outputs=x)
    return model
#===================
#Resnet50
#===================
def convolutional_block(x,filters,stride):
	shortcut = x
	f1,f2,f3 = filters
	x = Conv2D(f1,(1,1),padding='valid',strides=stride)(x)
	x = BatchNormalization(axis = 3)(x)
	x = Activation('relu')(x)
	x = Conv2D(f2, (3,3), padding='same',strides =1)(x)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(f3, (1, 1), padding='valid',strides = 1)(x)
	x = BatchNormalization(axis=3)(x)
	shortcut = Conv2D(f3,(1,1),padding = 'valid',strides = stride)(shortcut)
	shortcut = BatchNormalization(axis=3)(shortcut)
	x = Add()([x,shortcut])
	x = Activation('relu')(x)
	return x
def identity_block(x,filters):
	shortcut = x
	f1,f2,f3 = filters
	x = Conv2D(f1,(1,1),padding='valid')(x)
	x = BatchNormalization(axis=3)(x)
	x = Activation('relu')(x)
	x = Conv2D(f2,(3,3),padding='same')(x)
	x = BatchNormalization(axis = 3)(x)
	x = Activation('relu')(x)
	x = Conv2D(f3,(1,1),padding='valid')(x)
	x = BatchNormalization(axis=3)(x)
	x = Add()([x,shortcut])
	x = Activation('relu')(x)
	return x

def Resnet50(waferSizeX, waferSizeY):
    input = Input((waferSizeX, waferSizeY, 3))
    x = ZeroPadding2D((3,3))(input)
    x = Conv2D(64,(7,7),strides=(2,2))(x)
    x = BatchNormalization(axis = 3)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size =(3,3),strides=(2,2),padding='same')(x)
    x = convolutional_block(x,[64,64,256],stride=1)
    x = identity_block(x,[64,64,256])
    x = convolutional_block(x,[128,128,512],stride=1)
    x = identity_block(x,[128,128,512])
    x = convolutional_block(x,[256,256,1024],stride=2)
    x = identity_block(x,[256,256,1024])
    x = convolutional_block(x,[512,512,2048],stride=2 )
    x = identity_block(x,[512,512,2048])
    size = int(x.shape[1])
    x = AveragePooling2D(pool_size=(size, size))(x)
    x = Flatten()(x)
    x = Dense(8,activation='softmax')(x)
    model = Model(inputs = input,outputs= x)
    return model

#==================
#googLeNet V1
#==================
def inception_model(input, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj):

    conv_1x1 = Conv2D(filters=filters_1x1, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3_reduce = Conv2D(filters=filters_3x3_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_3x3 = Conv2D(filters=filters_3x3, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_3x3_reduce)
    conv_5x5_reduce  = Conv2D(filters=filters_5x5_reduce, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    conv_5x5 = Conv2D(filters=filters_5x5, kernel_size=(5, 5), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv_5x5_reduce)
    maxpool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), padding='same')(input)
    maxpool_proj = Conv2D(filters=filters_pool_proj, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool)
    inception_output = concatenate([conv_1x1, conv_3x3, conv_5x5, maxpool_proj], axis=3)  # use tf as backend
    return inception_output
#==================
def googLeNet(waferSizeX, waferSizeY):
#==================  
    weight_path = None
    input = Input(shape=(waferSizeX, waferSizeY, 3))
    conv1_7x7_s2 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same', activation='relu', kernel_regularizer=l2(0.01))(input)
    maxpool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1_7x7_s2)
    conv2_3x3_reduce = Conv2D(filters=64, kernel_size=(1, 1), padding='same', activation='relu', kernel_regularizer=l2(0.01))(maxpool1_3x3_s2)
    conv2_3x3 = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.01))(conv2_3x3_reduce)
    maxpool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2_3x3)
    inception_3a = inception_model(input=maxpool2_3x3_s2, filters_1x1=64, filters_3x3_reduce=96, filters_3x3=128, filters_5x5_reduce=16, filters_5x5=32, filters_pool_proj=32)
    inception_3b = inception_model(input=inception_3a, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=192, filters_5x5_reduce=32, filters_5x5=96, filters_pool_proj=64)
    maxpool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_3b)
    inception_4a = inception_model(input=maxpool3_3x3_s2, filters_1x1=192, filters_3x3_reduce=96, filters_3x3=208, filters_5x5_reduce=16, filters_5x5=48, filters_pool_proj=64)
    inception_4b = inception_model(input=inception_4a, filters_1x1=160, filters_3x3_reduce=112, filters_3x3=224, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4c = inception_model(input=inception_4b, filters_1x1=128, filters_3x3_reduce=128, filters_3x3=256, filters_5x5_reduce=24, filters_5x5=64, filters_pool_proj=64)
    inception_4d = inception_model(input=inception_4c, filters_1x1=112, filters_3x3_reduce=144, filters_3x3=288, filters_5x5_reduce=32, filters_5x5=64, filters_pool_proj=64)
    inception_4e = inception_model(input=inception_4d, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    maxpool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(inception_4e)
    inception_5a = inception_model(input=maxpool4_3x3_s2, filters_1x1=256, filters_3x3_reduce=160, filters_3x3=320, filters_5x5_reduce=32, filters_5x5=128, filters_pool_proj=128)
    inception_5b = inception_model(input=inception_5a, filters_1x1=384, filters_3x3_reduce=192, filters_3x3=384, filters_5x5_reduce=48, filters_5x5=128, filters_pool_proj=128)
    averagepool1_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(7, 7), padding='same')(inception_5b)
    drop1 = Dropout(rate=0.4)(averagepool1_7x7_s1)
    linear = Dense(units=8, activation='softmax', kernel_regularizer=l2(0.01))(Flatten()(drop1))
    last = linear
    model = Model(inputs=input, outputs=last)
    return model