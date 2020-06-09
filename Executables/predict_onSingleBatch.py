def predictOnSingleBatch(model_builded, datasetDir):
    import tensorflow as tf
    import os
    import glob
    import shutil
    import cv2
    import numpy as np
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix, classification_report

    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']

    labels = []
    predicted_labels = []
    for index in range (0, 6):
        images = glob.glob(datasetDir+"validation/*.jpg")
        # Predict phase
        i = 0
        for image in images:
            img = cv2.imread(image, 1)
            imgAux = tf.expand_dims(img, axis=0)
            Y_pred = model_builded.predict(imgAux)
            prediccion = np.argmax(Y_pred, axis=1)
            predicted_labels.append(prediccion)
            labels.append(index)

    matriz_confusion = confusion_matrix(labels, predicted_labels)
    print('Confusion Matrix')
    print(matriz_confusion)
    print('Accuracy: ', accuracy_score(labels, predicted_labels))
    print('Classification Report')
    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']
    print(classification_report(labels, predicted_labels, target_names=class_names))
