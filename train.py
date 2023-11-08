import cv2
from keras.applications.mobilenet import MobileNet
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model

from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.callbacks import ModelCheckpoint

n_class = 14

def get_model():
    base_model = MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(512, activation='relu')(x)
    outs = Dense(n_class, activation='softmax')(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.inputs, outputs= outs)
    return model


model = get_model()

data_folder = "data"

train_datagen = ImageDataGenerator(preprocessing_function=keras.applications.mobilenet.preprocess_input,
                                   rotation_range=0.2,
                                   width_shift_range=0.2, height_shift_range=0.2, shear_range=0.3, zoom_range=0.5,
                                   horizontal_flip=True, vertical_flip=True,
                                   validation_split=0.2)

train_generator = train_datagen.flow_from_directory(data_folder,
                                                    target_size=(224, 224),
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_folder,  # same directory as training data
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    subset='validation')

classes = train_generator.class_indices
print(classes)
classes = list(classes.keys())


def train_model():
    n_epochs = 30
    batch_size = 64
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('models/best.hdf5', monitor='val_loss', save_best_only = True, mode='auto')
    callback_list = [checkpoint]

    step_train = train_generator.n//batch_size
    step_val = validation_generator.n//batch_size

    model.fit(train_generator, steps_per_epoch=step_train,
                        validation_data=validation_generator,
                        validation_steps=step_val,
                        callbacks=callback_list,
                        epochs=n_epochs)

    model.save('models/model.h5')

if __name__ == '__main__':
    train_model()
