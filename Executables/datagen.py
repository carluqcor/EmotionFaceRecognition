def generateDatagen(datasetDir, batch_size):
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    train_dir = datasetDir+'training/'
    val_dir = datasetDir+'validation/'
    img_width = img_height = 224

    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        zoom_range=[0.8, 1.0],
        rotation_range=5,
        shear_range=0.1,
        width_shift_range=0.1,
        brightness_range=[0.75,1.1],
        data_format='channels_last')

    val_datagen = ImageDataGenerator(
        data_format='channels_last')

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')  # set as training data

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')  # set as validation data

    return train_generator, validation_generator
