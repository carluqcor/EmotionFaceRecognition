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
import argparse

# Operations to apply 
def AVG(ann, windowSize):
    annT = ann[0]
    for idx in range(1, windowSize):
        annT = annT + ann[idx]
    return annT / windowSize


def MEDIAN(ann, windowSize):
    return np.median(np.asarray(ann)[0:windowSize], axis=0)


def MAX(ann, windowSize):
    return np.max(np.asarray(ann)[0:windowSize], axis=0)


# Plot faces with a emotion bar
def plotEmotions(pred, framesCount, miniBatch, resultsDir):
    frame = cv2.cvtColor(miniBatch, cv2.COLOR_BGR2RGB)
    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']
    fig = plt.figure()
    fig.add_subplot(221)
    plt.title('Sample')
    plt.imshow(frame)

    y_pos = np.arange(len(class_names))

    fig.add_subplot(222)
    plt.title('Emotion')
    plt.barh(y_pos, pred[0], color='green')
    plt.yticks(y_pos, class_names)
    plt.ylabel('Emotions')
    plt.xlabel('Probability')
    plt.title('Predict')

    fig.tight_layout(pad=3.0)
    plt.savefig(resultsDir+str(framesCount)+'.png')
    plt.close()

if __name__ == "__main__":
    dirc = False
    camera = False
    video = False
    # Colors for emotions to plot
    color = [(255, 0, 0), (250, 32, 236), (35, 235, 13),
              (255, 147, 4), (4, 38, 255), (239, 255, 4)]
    
    class_names = ['An: ', 'Sc: ', 'Ha: ', 'Di: ', 'Sa: ', 'Su: ']
    
    # Default values
    operation = AVG
    windowSize = 1
    modelName = '../Models/vgg19.h5'
    detectorType = 2
    confidenceArg = 0.75
    faceDetector = cv2.dnn.readNetFromCaffe('../Models/deploy.prototxt.txt', '../Models/res10_300x300_ssd_iter_140000.caffemodel')

    parser = argparse.ArgumentParser(description='Executable to test Emotion Face Recognition.')
    parser.add_argument('-r', dest='resultsDir',
                        help='directory name to save results', required=True)
    parser.add_argument('-c', dest='confidence',
                        help='confidence for dnn OpenCV', default=0.75)
    parser.add_argument('-m', dest='model',
                        help='model file to load', default='../Models/vgg19.h5')
    parser.add_argument('-w', dest='windowSize',
                        help='window size for apply operations', default=3)
    parser.add_argument('-o', dest='operation',
                        help='operation to apply AVG|MAX|MEDIAN', default='AVG')
    parser.add_argument('-d', dest='detector',
                        help='detector to use on face detection OPENCV|DLIB|DNN', default='DNN')
    parser.add_argument('-i', dest='inputVar',
                        help='decide to use image, folder, camera or video', default='0')
    args = parser.parse_args()

    resultsDir = args.resultsDir
    if os.path.isdir(str(resultsDir)):
        raise Exception('Directory path provided already exist, please give different one or delete it')
    else:
        os.mkdir(resultsDir)

    if args.windowSize:
        windowSize = int(args.windowSize)
    if args.confidence:
        confidenceArg = float(args.confidence)
    if args.model:
        modelName = args.model
        model_builded = load_model(str(modelName))
    if args.operation:
        if args.operation == "avg" or args.operation == "AVG":
            operation = AVG
        elif args.operation == "median" or args.operation == "MEDIAN":
            operation = MEDIAN
        elif args.operation == "max" or args.operation == "MAX":
            operation = MAX
    if args.detector:
        if str(args.detector) == 'DLIB' or str(args.detector) == 'dlib':
            faceDetector = dlib.get_frontal_face_detector()
            detectorType = 0
        elif str(args.detector) == 'OPENCV' or str(args.detector) == 'opencv':
            faceDetector = cv2.CascadeClassifier('../Models/haarcascade_frontalface_default.xml')
            detectorType = 1
        elif str(args.detector) == 'DNN' or str(args.detector) == 'dnn':
            faceDetector = cv2.dnn.readNetFromCaffe('../Models/deploy.prototxt.txt', '../Models/res10_300x300_ssd_iter_140000.caffemodel')
            detectorType = 2
    if args.inputVar:
        if os.path.isdir(str(args.inputVar)):
            images = glob.glob(str(args.inputVar)+'*.jpg')
            dirc = True
            camera = False
        elif os.path.isfile(str(args.inputVar)) and '.mp4' not in str(args.inputVar):
            imageSolo = cv2.imread(str(args.inputVar), 1)
            dirc = False
            camera = False
        elif args.inputVar == '0':
            cap = cv2.VideoCapture(0)
            if not (cap.isOpened()):
                print(
                    '\033[91m'+"The device could not be opened for option -i."+'\033[0m')
                sys.exit(-1)
            else:
                camera = True
        else:
            cap = cv2.VideoCapture(str(args.inputVar))

            if not (cap.isOpened()):
                print('\033[91m'+'Argument is not valid for option -i, try ' +
                        '\033[93m'+'inputImage/Folder/CameraID/Video '+'\033[0m')
                sys.exit(-1)
            else:
                camera = True
                video = True
    
    # Class names
    class_names = ['An: ', 'Sc: ', 'Ha: ', 'Di: ', 'Sa: ', 'Su: ']

    framesCount = 0
    # Input: Camera
    if camera:
        # Temporal window to apply
        if windowSize > 1:
            ann = []
            index = 0
            # While there are frames
            while(True):
                ret, frame = cap.read()
                frameSaved = frame
                # Detect the faces
                # OpenCV face detector
                if detectorType == 1:
                    faces = faceDetector.detectMultiScale(frame, 1.2, 4)
                    facesLen = 0

                    k=cv2.waitKey(1)
                    if k == 27: # Escape
                        break
                    if k == 32: # Space to plot
                        indix=0
                        # Get detected faces
                        for (x, y, w, h) in faces:
                            crop_img=cv2.resize(
                                frameSaved[y:y+h, x:x+w], (224, 224))
                            imgAux=tf.expand_dims(crop_img, axis=0)
                            pred=model_builded.predict(imgAux)
                            plotEmotions(pred, str(
                                framesCount)+"_"+str(indix), crop_img, resultsDir)
                            indix=indix + 1
                        framesCount=framesCount + 1
                        
                    # Get detected faces
                    for (x, y, w, h) in faces:
                        auxPred = []
                        crop_img = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                        imgAux = tf.expand_dims(crop_img, axis=0)
                        pred = np.round(np.asarray(
                            model_builded.predict(imgAux)), decimals=3)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        try:
                            # First faces
                            if len(ann[facesLen]) == 0:
                                ann.append([])
                                ann[facesLen].append(pred[0])
                            # Clearing buffer
                            elif (len(ann[facesLen]) == windowSize):
                                ann[facesLen].pop(0)
                                ann[facesLen].append(pred[0])
                            # Push faces data until windowSize
                            elif len(ann[facesLen]) > 0:
                                ann[facesLen].append(pred[0])
                        # Handling first face
                        except IndexError:
                            ann.append([])
                            ann[facesLen].append(pred[0])
                        # Applying operation
                        if(windowSize == len(ann[facesLen])):
                            auxPred = np.round(np.asarray(
                                operation(ann[facesLen], windowSize)),
                                decimals=3)
                        else:
                            auxPred = np.round(np.asarray(pred[0]), decimals=3)
                        # To move vertically down when plotting classnames and their probs
                        printable_y=-10
                        # Ploting probs
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                auxPred[face]), (x, y-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15  
                        facesLen += 1
                    cv2.imshow('Emotion Detector', frame)
                # DLib face detector
                elif detectorType == 0:
                    dets=faceDetector(frame, 0)

                    k=cv2.waitKey(1)
                    if k == 27:
                        break
                    if k == 32:
                        indix=0
                        # Get faces detected
                        for i, d in enumerate(dets):
                            crop_img=cv2.resize(
                                frameSaved[d.top():d.top()+
                                d.bottom()-d.top(),
                                d.left():d.left()+
                                d.right()-d.left()
                            ],
                            (224, 224))
                            imgAux=tf.expand_dims(crop_img, axis=0)
                            pred=model_builded.predict(imgAux)
                            plotEmotions(pred, str(
                                framesCount)+"_"+str(indix), crop_img, resultsDir)
                            indix=indix + 1
                        framesCount=framesCount + 1

                    for i, d in enumerate(dets):
                        # Cropping faces from frame
                        crop_img=cv2.resize(
                            frameSaved[d.top():d.top()+
                            d.bottom()-d.top(),
                            d.left():d.left()+
                            d.right()-d.left()
                        ],
                        (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(imgAux)
                        cv2.rectangle(frame, (d.left(), 
                            d.top()), 
                            (d.left()+d.right()-d.left(), 
                            d.top()+d.bottom()-d.top()),
                            (255, 0, 0), 2)
                        try:
                            if len(ann[i]) == 0:
                                ann.append([])
                                ann[i].append(pred[0])
                            elif (len(ann[i]) == windowSize):
                                ann[i].pop(0)
                                ann[i].append(pred[0])
                            elif len(ann[i]) > 0:
                                ann[i].append(pred[0])
                        except IndexError:
                            ann.append([])
                            ann[i].append(pred[0])
                        if(windowSize == len(ann[i])):
                            auxPred = np.round(np.asarray(operation(
                                ann[i], windowSize)), 
                                decimals=3)
                        else:
                            auxPred = np.round(np.asarray(pred[0]), decimals=3)
                        printable_y=-10
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                auxPred[face]), (d.left(
                            ), d.top()-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15
                    cv2.imshow('Emotion Detector', frame)
                # OpenCV DNN face detector
                else:
                    try:
                        (h, w) = frameSaved.shape[:2]
                    except:
                        print('Finished')
                        sys.exit(1)
                    blob = cv2.dnn.blobFromImage(cv2.resize(
                        frameSaved, (224, 224)), 
                        1.0, (224, 224), 
                        (104.0, 177.0, 123.0))

                    faceDetector.setInput(blob)
                    detections = faceDetector.forward()
                    facesLen = 0

                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    if key == 32:
                        indix=0
                        for i in range(0, detections.shape[2]):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            confidence = detections[0, 0, i, 2]
                            
                            # Check detection confidence
                            if (confidence > confidenceArg):
                                crop_img=cv2.resize(
                                    frame[startY:endY, startX-10:endX+10], (224, 224))
                                imgAux=tf.expand_dims(crop_img, axis=0)
                                pred=model_builded.predict(imgAux)
                                plotEmotions(pred, str(
                                    framesCount)+ "_" +str(indix), crop_img, resultsDir)
                                indix=indix + 1
                        framesCount=framesCount + 1

                    for i in range(0, detections.shape[2]):
                        if (detections[0, 0, i, 2] > confidenceArg):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            try:
                                cv2.rectangle(frameSaved, (startX-10, startY), (endX+10, endY), (255, 255, 255), 2)
                                auxPred = []
                                crop_img = cv2.resize(frameSaved [startY:endY, startX-10:endX+10], (224, 224))
                                imgAux = tf.expand_dims(crop_img, axis=0)
                                pred = np.round(np.asarray( model_builded.predict(imgAux)), 
                                    decimals=3)
                                try:
                                    if len(ann[facesLen]) == 0:
                                        ann.append([])
                                        ann[facesLen].append( pred[0])
                                    elif (len(ann[facesLen]) == windowSize):
                                        ann[facesLen].pop(0)
                                        ann[facesLen].append( pred[0])
                                    elif len(ann[facesLen]) > 0:
                                        ann[facesLen].append( pred[0])
                                except IndexError:
                                    ann.append([])
                                    ann[facesLen].append(pred[0])
                                if(windowSize == len(ann[facesLen])):
                                    auxPred = np.round(np.asarray( operation(ann[facesLen],
                                    windowSize)), decimals=3)
                                else:
                                    auxPred = np.round(np.asarray( pred[0]), decimals=3)
                                printable_y = -10
                                for face in range(0, 6):
                                    cv2.putText(frameSaved, str(class_names[face]) + ': ' + str(
                                        auxPred[face]), (startX, startY-printable_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, color[face], 2)
                                    printable_y=printable_y - 15  
                                facesLen += 1
                            except cv2.error as e:
                                break

                    cv2.imshow('Emotion Detector', frameSaved)

        # No temporal window
        else:
            ann = []
            index = 0
            while(True):
                ret, frame = cap.read()
                frameSaved = frame
                # OpenCV face detector
                if detectorType == 1:
                    faces = faceDetector.detectMultiScale(frame, 1.2, 4)
                    print(faces)
                    facesLen = 0
                    k=cv2.waitKey(1)
                    if k == 27:
                        break
                    if k == 32:
                        indix=0
                        for (x, y, w, h) in faces:
                            crop_img=cv2.resize(
                                frameSaved[y:y+h, x:x+w], (224, 224))
                            imgAux=tf.expand_dims(crop_img, axis=0)
                            pred=model_builded.predict(imgAux)
                            plotEmotions(pred, str(
                                framesCount)+"_"+str(indix), crop_img, resultsDir)
                            indix=indix + 1

                        framesCount=framesCount + 1
                    for (x, y, w, h) in faces:
                        auxPred = []
                        crop_img = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                        imgAux = tf.expand_dims(crop_img, axis=0)
                        pred = np.round(np.asarray( model_builded.predict(imgAux)), decimals=3)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        printable_y=-10
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                pred[0][face]), (x, y-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15  
                        facesLen += 1
                    cv2.imshow('Emotion Detector', frame)
                # DLib face detector
                elif detectorType == 0:
                    dets=faceDetector(frame, 0)
                    k=cv2.waitKey(1)
                    if k == 27:
                        break
                    if k == 32:
                        indix=0
                        for i, d in enumerate(dets):
                            crop_img=cv2.resize(
                                frameSaved[d.top():d.top()+
                                d.bottom()-d.top(),
                                d.left():d.left()+
                                d.right()-d.left()
                            ], (224, 224))
                            imgAux=tf.expand_dims(crop_img, axis=0)
                            pred=model_builded.predict(imgAux)
                            plotEmotions(pred, str(
                            framesCount)+"_"+str(indix), crop_img, resultsDir)
                            indix=indix + 1
                        framesCount=framesCount + 1

                    for i, d in enumerate(dets):
                        crop_img=cv2.resize(
                                frameSaved[d.top():d.top()+
                                d.bottom()-d.top(),
                                d.left():d.left()+
                                d.right()-d.left()
                            ], (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=np.round(np.asarray(model_builded.predict(imgAux)), decimals=3)
                        cv2.rectangle(frame, (d.left(), d.top()), (d.left()+d.right()-d.left(), d.top()+d.bottom()-d.top()), (255, 0, 0), 2)
                        printable_y=-10
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                pred[0][face]), (d.left(
                            ), d.top()-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15
                    cv2.imshow('Emotion Detector', frame)
                # DNN face detector
                else:
                    (h, w) = frameSaved.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(
                        frameSaved, (224, 224)), 
                        1.0, (224, 224), 
                        (104.0, 177.0, 123.0))

                    faceDetector.setInput(blob)
                    detections = faceDetector.forward()
                    facesLen = 0

                    key = cv2.waitKey(1)
                    if key == 27:
                        break
                    if key == 32:
                        indix=0
                        for i in range(0, detections.shape[2]):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            if (detections[0, 0, i, 2] > confidenceArg):
                                crop_img=cv2.resize(
                                    frame[startY:endY, startX-10:endX+10], (224, 224))
                                imgAux=tf.expand_dims(crop_img, axis=0)
                                pred=model_builded.predict(imgAux)
                                plotEmotions(pred, str(
                                    framesCount) + "_" + str(indix), crop_img, resultsDir)

                                indix=indix + 1
                        framesCount=framesCount + 1

                    for i in range(0, detections.shape[2]):
                        if (detections[0, 0, i, 2] > confidenceArg):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            try:
                                cv2.rectangle(frameSaved, (startX-10, startY), (endX+10, endY), (255, 255, 255), 2)
                                crop_img = cv2.resize( frameSaved[startY:endY, 
                                    startX-10:endX+10], (224, 224))
                                imgAux = tf.expand_dims(crop_img, axis=0)
                                pred = np.round(np.asarray( model_builded.predict(imgAux)),
                                    decimals=3)
                                printable_y = -10
                                for face in range(0, 6):
                                    cv2.putText(frameSaved, str(class_names[face]) + ': ' + str(
                                        pred[0][face]), (startX, startY-printable_y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.5, color[face], 2)
                                    printable_y=printable_y - 15  
                                facesLen += 1
                            except cv2.error as e:
                                break
                    cv2.imshow('Emotion Detector', frameSaved)
        # Destroy opencv windows
        cap.release()
        cv2.destroyAllWindows()
    # Input: Directory 
    if dirc:
        framesCount=0
        for image in images:
            imgAux=cv2.imread(image, 1)
            
            if detectorType == 1:
                faces=faceDetector.detectMultiScale(imgAux, 1.1, 4)
                index=0
                for (x, y, w, h) in faces:
                    crop_img=cv2.resize(imgAux[y:y+h, x:x+w], (224, 224))
                    expanded=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(expanded)

                    plotEmotions(pred, str(framesCount) + "_" +str(index), crop_img, resultsDir)
                    index=index + 1
                framesCount=framesCount + 1
            # DLib face detector
            elif detectorType == 0:
                dets=faceDetector(imgAux, 0)
                index=0
                for i, d in enumerate(dets):
                    crop_img=cv2.resize(imgAux[d.top():d.top(
                    )+d.bottom()-d.top(), d.left():d.left()+d.right()-d.left()], (224, 224))
                    expanded=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(expanded)

                    plotEmotions(pred, str(framesCount) + "_" +str(index), crop_img, resultsDir)
                    index=index + 1
            # DNN OpenCV face detector
            else:
                (h, w) = imgAux.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(imgAux, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

                faceDetector.setInput(blob)
                detections = faceDetector.forward()
                index=0
                for i in range(0, detections.shape[2]):
                    if (detections[0, 0, i, 2] > 0.6):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        crop_img=cv2.resize(imgAux[startY:endY, startX:endX], (224, 224))
                        expanded = tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(expanded)

                        plotEmotions(pred, str(framesCount)+'_'+str(index), crop_img, resultsDir)
                        index=index + 1
                framesCount=framesCount + 1
    # Input: Image
    else:
        try:
            # OpenCV face detector
            if detectorType == 1:
                faces = faceDetector.detectMultiScale(imageSolo, 1.1, 4)
                index=0
                for (x, y, w, h) in faces:
                    crop_img=cv2.resize(
                        imageSolo[y:y+h, x:x+w], (224, 224))
                    imgAux=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(imgAux)

                    plotEmotions(pred, str(index), crop_img, resultsDir)
                    index=index + 1
            # DLib face detector
            elif detectorType == 0:
                dets=faceDetector(imageSolo, 0)
                if len(dets) is not 0:
                    index=0
                    for i, d in enumerate(dets):
                        crop_img=cv2.resize(
                                imageSolo[d.top():d.top()+
                                d.bottom()-d.top(),
                                d.left():d.left()+
                                d.right()-d.left()
                            ], (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(imgAux)
                        plotEmotions(pred, str(index), crop_img, resultsDir)
                        index=index + 1
            # DNN OpenCV face detector
            else:
                (h, w) = imageSolo.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(imageSolo, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

                faceDetector.setInput(blob)
                detections = faceDetector.forward()
                index=0
                for i in range(0, detections.shape[2]):
                    if (detections[0, 0, i, 2] > 0.6):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        crop_img=cv2.resize(imageSolo [startY:endY, startX:endX], (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(imgAux)

                        plotEmotions(pred, str(index), crop_img, resultsDir)
                        index=index + 1
        # Handling error when escaping camera
        except NameError:
            sys.exit(0)
