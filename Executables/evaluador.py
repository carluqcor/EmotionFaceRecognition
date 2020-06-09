import os
import sys
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import dlib
import tensorflow as tf
import pylab as plt
import glob
import operator
import getopt
import argparse


def AVG(headTags, windowSize):
    headTagsT = headTags[0]
    for idx in range(1, windowSize):
        headTagsT = headTagsT + headTags[idx]
    return headTagsT / windowSize


def MEDIAN(headTags, windowSize):
    return np.median(np.asarray(headTags)[0:windowSize], axis=0)


def MAX(headTags, windowSize):
    return np.max(np.asarray(headTags)[0:windowSize], axis=0)

# Función para definir la puntuación a obtener según las emociones etiquetadas a cada cara
def score(lenEmotion):
    if (lenEmotion == 1):
        return [1.0, 0.0]
    elif (lenEmotion == 2):
        return [0.75, 0.25]
    elif (lenEmotion == 3):
        return [0.5, 0.25, 0.25]
    elif (lenEmotion == 4):
        return [0.4, 0.2, 0.2, 0.2]


if __name__ == "__main__":   
    model = load_model('../Models/vgg19.h5')
    windowSize = 3
    operation = 'AVG'
    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']

    parser = argparse.ArgumentParser(description='Evaluate labeled frames.')
    parser.add_argument('-f', dest='headFile',
                        help='head annotation file', required=True)
    parser.add_argument('-v', dest='videoFile',
                        help='video file', required=True)
    parser.add_argument('-m', dest='model',
                        help='model file')
    parser.add_argument('-w', dest='windowSize',
                        help='window size for apply operations')
    parser.add_argument('-o', dest='operation',
                        help='operation to apply AVG|MAX|MEDIAN')
    args = parser.parse_args()

    if args.headFile:  
        try:
            f = open(str(args.headFile), "r")
        except IOError:
            raise Exception('\033[91m'+'Annotation file not found'+'\033[0m')    
    if args.videoFile:  
        cap = cv2.VideoCapture(str(args.videoFile))
        if not (cap.isOpened()):
            raise Exception('\033[91m'+'Video not found'+'\033[0m')
    if args.model:
        if(os.path.isfile(str(args.model))):
            model = load_model(str(args.model))
        else:
            raise Exception('\033[91m'+'Model file not found'+'\033[0m')
    if args.windowSize:
        windowSize = int(args.windowSize)
    if args.operation:
        if str(args.operation) == "avg" or str(args.operation) == "AVG":
            operation = AVG
        elif str(args.operation) == "median" or str(args.operation) == "MEDIAN":
            operation = MEDIAN
        elif str(args.operation) == "max" or str(args.operation) == "MAX":
            operation = MAX
        else:
            raise Exception('\033[91m'+'Operation '+'\033[93m'+args.operation+'\033[91m' +
                    ' is not available for option -o, try '+'\033[93m'+' AVG, MAX o MEDIAN'+'\033[0m')
    f1 = f.read().splitlines()
    frameIndex = 1
    linesNum = 0
    totalToScore = 0
    scored = 0
    headTags = []
    index = 0
    while(True):
        ret, frame = cap.read()
        if (not ret):
            print('Finished evaluating!')
            break
        else:
            if windowSize > 1:
                facesLen = 0
                print('\nFrame: ',frameIndex)
                linesNum += 1
                try:
                    # Mientras siga siendo el mismo frame
                    while(len(f1[linesNum].split(' ')) != 1):
                        if ((f1[linesNum].split(' ')[4]) == 'Skip'):
                            linesNum += 1
                        if(len(f1[linesNum].split(' ')) == 1):
                            linesNum += 1
                        # Cosas de la red neuronal (predict)
                        crop_img = cv2.resize(frame[int(f1[linesNum].split(' ')[1]):int(f1[linesNum].split(' ')[3]), int(f1[linesNum].split(' ')[0]):int(f1[linesNum].split(' ')[2])], (224, 224))
                        imgAux = tf.expand_dims(crop_img, axis=0)
                        Y_pred = model.predict(imgAux)

                        try:
                            if len(headTags[facesLen]) == 0:  # Resto de primeras caras
                                headTags.append([])
                                headTags[facesLen].append(Y_pred[0])
                            # Para que limpie el buffer
                            elif (len(headTags[facesLen]) == windowSize):
                                headTags[facesLen].pop(0)
                                headTags[facesLen].append(Y_pred[0])
                            # Para rellenar el frame hasta el windowSize frame de cada cara
                            elif len(headTags[facesLen]) > 0:
                                headTags[facesLen].append(Y_pred[0])
                        except IndexError:  # Primera cara
                            headTags.append([])
                            headTags[facesLen].append(Y_pred[0])
                        if(windowSize == len(headTags[facesLen])):  # Calculo de la operación
                            auxPred = np.asarray(operation(headTags[facesLen], windowSize))
                        else:
                            auxPred = np.asarray(Y_pred[0])
                        # Obtiene las emociones etiquetadas
                        emotions = f1[linesNum].split(' ')[5:len(f1[linesNum].split(' '))]

                        # Calcula el score a usar y se lo headTags al total para la confianza
                        puntuacion = score(len(emotions))
                        totalToScore += sum(puntuacion)
                        
                        # Aquí obtengo un vector de indices según si es el mayor sentimiento
                        # predicho hasta el menor
                        print('Predictions: ',np.round(auxPred, decimals = 3))
                        print('Classes: ', class_names)
                        print('Emotions labeled: ', emotions)
                        indices = sorted(range(len(auxPred)), key=lambda i: auxPred[i], reverse=True)[:len(emotions)]             

                        # Comprueba si las emociones coinciden
                        # La primera emoción siempre va a ser más puntuada porque será la que
                        # más alta esté en el vector de predicciones
                        for indice in range(0, len(indices)):
                            if(class_names[indices[0]] in emotions and indice == 0):
                                scored += puntuacion[0]
                            elif (indice != 0 and class_names[indices[indice]] in emotions):
                                scored += puntuacion[1]
                        print('Scored ', np.round(scored, decimals = 2), ' at frame: ', frameIndex)
                        linesNum += 1
                except IndexError:
                    f.close()
                    cap.release()
            else:
                print('\nFrame: ',frameIndex)
                linesNum += 1
                try:
                    # Mientras siga siendo el mismo frame
                    while(len(f1[linesNum].split(' ')) != 1):
                        if(len(f1[linesNum].split(' ')) == 1):
                            linesNum += 1
                        # Cosas de la red neuronal (predict)
                        crop_img = cv2.resize(frame[int(f1[linesNum].split(' ')[1]):int(f1[linesNum].split(' ')[3]), int(f1[linesNum].split(' ')[0]):int(f1[linesNum].split(' ')[2])], (224, 224))
                        imgAux = tf.expand_dims(crop_img, axis=0)
                        Y_pred = model.predict(imgAux)

                        # Obtiene las emociones etiquetadas
                        emotions = f1[linesNum].split(' ')[5:len(f1[linesNum].split(' '))]

                        # Calcula el score a usar y se lo headTags al total para la confianza
                        puntuacion = score(len(emotions))
                        totalToScore += sum(puntuacion)
                        
                        # Aquí obtengo un vector de indices según si es el mayor sentimiento
                        # predicho hasta el menor
                        print(np.round(Y_pred[0], decimals = 3))
                        print(class_names)
                        print(emotions)
                        indices = sorted(range(len(Y_pred[0])), key=lambda i: Y_pred[0][i], reverse=True)[:len(emotions)]

                        # Comprueba si las emociones coinciden
                        # La primera emoción siempre va a ser más puntuada porque será la que
                        # más alta esté en el vector de predicciones
                        for indice in range(0, len(indices)):
                            if(class_names[indices[0]] in emotions and indice == 0):
                                scored += puntuacion[0]
                            elif (indice != 0 and class_names[indices[indice]] in emotions):
                                scored += puntuacion[1]
                        print('Scored ', np.round(scored, decimals = 2), ' at frame: ', frameIndex)
                        linesNum += 1
                except IndexError:
                    f.close()
                    cap.release()
            frameIndex += 1

    # Puntuaciones
    print('\n\n\n\n\nTotal a obtener: ', totalToScore)
    print('Obtenido: ', np.round(scored, decimals=2))
    print('Confidence: ', np.round(scored/totalToScore, decimals=2))
    sys.exit(1)
