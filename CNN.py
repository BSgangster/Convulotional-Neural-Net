"""
Warning was compiled and tested using jupyter, however it is completely possible to compile this code without jypter,
just simply run it with command >> python3 CNN.py

However to train the model again remove the triple quotes comments from the section that says '#CNN training block'
After this I suggest you comment out the section that says '#After training'
Afer you have excuted '$CNN training block' you can now add back '#After training' and comment out "#CNN training block" and the CNN will -
now identify two different objects from image files related to the objects.
"""

"""
#CNN training block
#Importing the Keras libs and packs
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initializing the cnn
classifier = Sequential()

#adding the convu layer
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))

#adding the pooling layer
classifier.add(MaxPooling2D(pool_size = (2,2)))

#adding second convu and max pooling
classifier.add(Convolution2D(32,(3,3),input_shape = (64,64,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))

#flattening layer
classifier.add(Flatten())

#connecting the cnn with ann completing the classification
classifier.add(Dense(units = 128,activation = 'relu'))
classifier.add(Dense(units = 1,activation = 'sigmoid'))

#compiling the cnn
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy',metrics = ['accuracy'])

#fitting the images using keras
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary')

validation_gen = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size = (64,64),
        batch_size = 32,
        class_mode = 'binary')

classifier.fit_generator(
        train_gen,
        steps_per_epoch=250,
        epochs = 25,
        validation_data = validation_gen,
        validation_steps = 2000)

import numpy as np
from keras.preprocessing import image
test_img = image.load_img('animal3.png',target_size = (64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
result = classifier.predict(test_img)
train_gen.class_indices
if result[0][0] > 0.5:
    print("dog")
else:
    print("cat")

classifier.save('my_model.h5')

"""

#After training
#Warning this code must only be added after model was creating.
from keras.models import load_model
classfier  = load_model('\hello.h5')
 
import numpy as np
from keras.preprocessing import image
test_img = image.load_img('F:\animal.png',target_size = (64,64))
test_img = image.img_to_array(test_img)
test_img = np.expand_dims(test_img, axis = 0)
result = classfier.predict(test_img)
#train_gen.class_indices
if result[0][0] > 0.5:
    print("dog")
else:
    print("cat")
