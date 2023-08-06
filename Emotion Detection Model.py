import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from matplotlib import pyplot as plt
import numpy as np

IMG_HEIGHT = 48
IMG_WIDTH = 48
BATCH_SIZE = 32

train_data_dir = 'data/train/'
test_data_dir = 'data/test/'

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   shear_range= 0.3,
                                   zoom_range=0.3,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                    color_mode='grayscale',
                                                    target_size = (IMG_HEIGHT,IMG_WIDTH),
                                                    batch_size=BATCH_SIZE,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(test_data_dir,
                                                              color_mode='grayscale',
                                                              target_size = (IMG_HEIGHT,IMG_WIDTH),
                                                              batch_size =BATCH_SIZE,
                                                              class_mode='categorical',
                                                              shuffle=True)

class_label=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
img, label = train_generator.__next__()

import random

i = random.randint(0, (img.shape[0])-1)
image = img[i]
labl = class_label[label[i].argmax()]
plt.imshow(image[:,:,0], cmap='gray')
plt.title(labl)
plt.show()

##################################################################
#MAKING MODEL

model = Sequential()

model.add(Conv2D(32, kernel_size = (3,3), activation='relu' , input_shape = (48,48,1)))

model.add(Conv2D(64, kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(126, kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size = (3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='relu'))

model.compile( optimizer = 'adam', loss='categorical_crossentropy', metrics = ['accuracy'])
     

#####################################################################
# TRAININ MODEL

train_path = "data/train/"
test_path = "data/test"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path): 
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)
    
epochs_1 = 250


history = model.fit(train_generator,
                    steps_per_epoch = num_train_imgs//BATCH_SIZE,
                    epochs= epochs_1,
                    validation_data = validation_generator,
                    validation_steps = num_test_imgs//BATCH_SIZE)

model.save('emotion_detection_model_250epochs.h5')

#######################################################################
# LOADING MODEL

from keras.models import load_model

my_model = load_model('emotion_detection_model_250epochs.h5', compile= False)

test_img , test_lbl = validation_generator.__next__()
predictions = my_model.predict(test_img)

predictions = np.argmax(predictions, axis=1)
test_labels = np.argmax(test_lbl, axis=1)

from sklearn import metrics
print("Accuracy = ", metrics.accuracy_score(test_labels,predictions))

#Condfusion Matrix - verify accuracy of each class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(test_labels, predictions)

import seaborn as sns
sns.heatmap(cm, annot = True)

################################################################
# TESTING MODEL

class_label=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']
#Check Result on few images
n = random.randint(0, test_img.shape[0]-1)
image= test_img[n]
orig_labl = class_label[test_labels[n]]
pred_labl = class_label[predictions[n]]
plt.imshow(image[:,:,0], cmap='gray')
plt.title("Original Label is:" + orig_labl + "Predicted label is:"+ pred_labl)
plt.show()