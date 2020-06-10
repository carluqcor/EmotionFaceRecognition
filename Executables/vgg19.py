if __name__ == "__main__":

    import tensorflow as tf
    
    #### Add GPU utils #### 

    from datagen import generateDatagen
    from sklearn.metrics import accuracy_score
    from predict_onSingleBatch import *
    from tensorflow.keras import regularizers
    from collections import Counter

    from imports import *
    import argparse
    
    # Parser

    parser = argparse.ArgumentParser(description='Executable for training VGG19 model.')
    parser.add_argument('-c', dest='modelCheckpointDir',
                        help='save model when val_loss improve')
    parser.add_argument('-b', dest='batchSize',
                        help='set batch size value', default=32)
    parser.add_argument('-d', dest='datasetDir',
                        help='path to dataset', required=True)
    parser.add_argument('-t', dest='tensorboardDir',
                        help='path to save tensorboard')
    parser.add_argument('-m', dest='modelName',
                        help='add model filename to be saved', required=True)
    parser.add_argument('-e', dest='epoch',
                        help='set epoch number', default=15)
    parser.add_argument('-f', dest='historyName',
                        help='add history filename to save history object after training')
    parser.add_argument('-n', dest='classes',
                        help='set number of classes', default=6)
    args = parser.parse_args()

    ModelName = args.modelName
    datasetDir = args.datasetDir

    callbacks = []
    
    if args.classes:
        n_classes = int(args.classes)
    if args.tensorboardDir:
        tensorboardDir = args.tensorboardDir
        if not os.path.isdir(tensorboardDir):
            os.mkdir(tensorboardDir)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboardDir)
        callbacks.append(tensorboard_callback)

    if args.batchSize:
        batchSize = int(args.batchSize)

    if args.epoch:
        epochs = int(args.epoch)

    if args.modelCheckpointDir:
        modelCheckpointDir = args.modelCheckpointDir
        checkpoint = ModelCheckpoint(filepath=modelCheckpointDir,
            save_weights_only=False,
            monitor='val_loss', 
            verbose=1, 
            save_best_only=True, 
            mode='min')
        callbacks.append(checkpoint)
        
    if args.historyName:
        historyName = args.historyName

    # If layer is added must change last frozen layer to first dense layer after convs blocks
    base_model = vgg19.VGG19(
            weights='imagenet',
            include_top=False,
            input_shape=(224, 224, 3))
 
    x = base_model.output

    x = GlobalAveragePooling2D()(x)

    x = Dense(units=704,
              activation='relu')(x)

    x = Dropout(0.2)(x)

    x = Dense(units=512,
              activation='relu')(x)

    x = Dropout(0.5)(x)

    salida = Dense(6,
                   activation='softmax')(x)

    model_builded = Model(inputs=base_model.input,
                          outputs=salida)

    # Compilamos el modelo
    lr_decayed = tf.keras.experimental.CosineDecayRestarts(1e-05, 1000, t_mul=2.0, m_mul=1.0, alpha=0.0, name=None)

    model_builded.compile(loss = tf.losses.CategoricalCrossentropy(label_smoothing=0.1),
                          optimizer=Adam(lr_decayed),
                          metrics=['acc'])
                          
    for convLayer in model_builded.layers[::-1]:
        if isinstance(convLayer, Conv2D):
            convLayer.trainable = False
  
    # Change if layer is added to first dense layer after convs blocks
    model_builded.layers[23].trainable = False
    
    # Class type
    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']

    # Add dataset generator from flow from directory below
    train_generator, validation_generator = generateDatagen(datasetDir, batchSize)

    # Early stopping
    earlystop = EarlyStopping(monitor='val_loss',
        mode='min',
        min_delta=0,
        patience=4,
        restore_best_weights=True,
        verbose=1)
    callbacks.append(earlystop)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        factor=0.2,
        cooldown=0,
        patience=5,
        min_lr=0,
        mode='min',
        epsilon=0.001,
        verbose=1)    
    callbacks.append(reduce_lr)


    # Entrenamiento del modelo
    history = model_builded.fit(train_generator,
        steps_per_epoch=train_generator.samples // batchSize,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batchSize,
        epochs=epochs,
        callbacks=callbacks)
    
    predictOnSingleBatch(model_builded)

    if historyName:
        with open(historyName, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        
    model_builded.save(ModelName+".h5")

    print('--- ACC (train):')
    print("Máximo:", max(np.array(history.history['acc'])))
    print("Mínimo:", min(np.array(history.history['acc'])))
    print("Media:", np.mean(np.array(history.history['acc'])))
    print("Desv. tipica:", np.std(np.array(history.history['acc'])))
    print('--- ACC (val):')
    print("Máximo:", max(np.array(history.history['val_acc'])))
    print("Mínimo:", min(np.array(history.history['val_acc'])))
    print("Media:", np.mean(np.array(history.history['val_acc'])))
    print("Desv. tipica:", np.std(np.array(history.history['val_acc'])))
