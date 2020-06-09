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
def plotEmotions(pred, framesCount, miniBatch):
    frame = cv2.cvtColor(miniBatch, cv2.COLOR_BGR2RGB)
    class_names = ['Angry', 'Scared', 'Happy', 'Disgusted', 'Sad', 'Surprised']
    fig = plt.figure()
    fig.add_subplot(221)
    plt.title('Muestra')
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
    plt.savefig(str(framesCount)+'.png')


if __name__ == "__main__":
    dirc = False
    camera = False
    video = False
    # Colors for emotions to plot
    color = [(36, 255, 12), (255, 45, 0), (251, 255, 0),
              (0, 89, 255), (216, 0, 255), (255, 0, 185)]

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:d:i:m:w:o:h", [
                                   "input=", "model=", "windowSize=", "operation=", "descriptor=", "confidence=", "help"])
    except getopt.GetoptError as e:
        print('\033[91m'+str(e))
        sys.exit(-1)

    # Default values
    operation = AVG
    windowSize = 1
    modelName = '../Models/vgg19.h5'
    descriptorType = 2
    confidenceArg = 0.75
    faceDescriptor = cv2.dnn.readNetFromCaffe('../Models/deploy.prototxt.txt', '../Models/res10_300x300_ssd_iter_140000.caffemodel')

    for opt, arg in opts:
        if opt in ('-w', '--windowSize'):
            windowSize = int(arg)
        elif opt in ('-h', '--help'):
            raise Exception('\033[93m'+'Usage: <demo.py> '+'\033[91m'+'\n\t-m modelFile \n\t-w windowSize \n\t-h help \n\t-i inputImage/Folder/CameraID/Video \n\t-o AVG/MAX/MEDIAN' +  '\n\t-c Float number'+ '\n\t-d OPENCV/DLIB/DNN \n'
                  '\033[92m'+'Example: python3.7 demo.py -i 0 -m models/vgg19_8.h5 -w 3 -o AVG -d OPENCV -c 0.8'+'\033[0m')
            sys.exit(0)
        elif opt in ('-d', '--descriptor'):
            if str(arg) == 'DLIB' or str(arg) == 'dlib':
                faceDescriptor = dlib.get_frontal_face_detector()
                descriptorType = 0
            elif str(arg) == 'OPENCV' or str(arg) == 'opencv':
                faceDescriptor = cv2.CascadeClassifier('../Models/haarcascade_frontalface_default.xml')
                descriptorType = 1
        elif opt in ('-c', '--confidence'):
            confidenceArg = float(arg)
        elif opt in ('-i', '--input'):
            if os.path.isdir(str(arg)):
                images = glob.glob(str(arg)+'*.jpg')
                dirc = True
                camera = False
            elif os.path.isfile(str(arg)) and '.mp4' not in str(arg):
                imageSolo = cv2.imread(str(arg), 1)
                dirc = False
                camera = False
            elif arg == '0':
                cap = cv2.VideoCapture(0)
                if not (cap.isOpened()):
                    print(
                        '\033[91m'+"The device could not be opened for option -i."+'\033[0m')
                    sys.exit(-1)
                else:
                    camera = True
            else:
                cap = cv2.VideoCapture(str(arg))

                if not (cap.isOpened()):
                    print('\033[91m'+'Argument is not valid for option -i, try ' +
                          '\033[93m'+'inputImage/Folder/CameraID/Video '+'\033[0m')
                    sys.exit(-1)
                else:
                    camera = True
                    video = True
        elif opt in ('-m', '--model'):
            model_builded = load_model(str(arg))
            print("Model loaded")
        elif opt in ('-o', '--operation'):
            if arg == "avg" or arg == "AVG":
                operation = AVG
            elif arg == "median" or arg == "MEDIAN":
                operation = MEDIAN
            elif arg == "max" or arg == "MAX":
                operation = MAX
            else:
                print('\033[91m'+'Operation '+'\033[93m'+arg+'\033[91m' +
                      ' is not available for option -o, try '+'\033[93m'+' AVG, MAX o MEDIAN'+'\033[0m')
                sys.exit(-1)
    
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
                # OpenCV face descriptor
                if descriptorType == 1:
                    faces = faceDescriptor.detectMultiScale(frame, 1.2, 4)
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
                                framesCount)+"_"+str(indix), crop_img)
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
                        printable_y=15
                        # Ploting probs
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                auxPred[face]), (x, y-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15  
                        facesLen += 1
                    cv2.imshow('Emotion Detector', frame)
                # DLib face descriptor
                elif descriptorType == 0:
                    dets=faceDescriptor(frame, 0)

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
                                framesCount)+"_"+str(indix), crop_img)
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
                        printable_y=15
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                auxPred[face]), (d.left(
                            ), d.top()-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15
                    cv2.imshow('Emotion Detector', frame)
                # OpenCV DNN face detector
                else:
                    (h, w) = frameSaved.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(
                        frameSaved, (224, 224)), 
                        1.0, (224, 224), 
                        (104.0, 177.0, 123.0))

                    faceDescriptor.setInput(blob)
                    detections = faceDescriptor.forward()
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
                                    framesCount)+ "_" +str(indix), crop_img)
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
                                printable_y = 15
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
                # OpenCV face descriptor
                if descriptorType == 1:
                    faces = faceDescriptor.detectMultiScale(frame, 1.2, 4)
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
                                framesCount)+"_"+str(indix), crop_img)
                            indix=indix + 1

                        framesCount=framesCount + 1
                    for (x, y, w, h) in faces:
                        auxPred = []
                        crop_img = cv2.resize(frame[y:y+h, x:x+w], (224, 224))
                        imgAux = tf.expand_dims(crop_img, axis=0)
                        pred = np.round(np.asarray( model_builded.predict(imgAux)), decimals=3)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        printable_y=15
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                pred[0][face]), (x, y-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15  
                        facesLen += 1
                    cv2.imshow('Emotion Detector', frame)
                # DLib face descriptor
                elif descriptorType == 0:
                    dets=faceDescriptor(frame, 0)
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
                            framesCount)+"_"+str(indix), crop_img)
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
                        pred=model_builded.predict(imgAux)
                        cv2.rectangle(frame, (d.left(), d.top()), (d.left()+d.right()-d.left(), d.top()+d.bottom()-d.top()), (255, 0, 0), 2)
                        printable_y=15
                        for face in range(0, 6):
                            cv2.putText(frame, str(class_names[face])+': '+str(
                                pred[0][face]), (d.left(
                            ), d.top()-printable_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color[face], 2)
                            printable_y=printable_y - 15
                    cv2.imshow('Emotion Detector', frame)
                # DNN face descriptor
                else:
                    (h, w) = frameSaved.shape[:2]
                    blob = cv2.dnn.blobFromImage(cv2.resize(
                        frameSaved, (224, 224)), 
                        1.0, (224, 224), 
                        (104.0, 177.0, 123.0))

                    faceDescriptor.setInput(blob)
                    detections = faceDescriptor.forward()
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
                                    framesCount) + "_" + str(indix), crop_img)
                                indix=indix + 1
                        framesCount=framesCount + 1

                    for i in range(0, detections.shape[2]):
                        if (detections[0, 0, i, 2] > confidenceArg):
                            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                            (startX, startY, endX, endY) = box.astype("int")
                            try:
                                cv2.rectangle(frameSaved, (startX-10, startY), (endX+10, endY), (255, 255, 255), 2)
                                auxPred = []
                                crop_img = cv2.resize( frameSaved[startY:endY, 
                                    startX-10:endX+10], (224, 224))
                                imgAux = tf.expand_dims(crop_img, axis=0)
                                pred = np.round(np.asarray( model_builded.predict(imgAux)),
                                    decimals=3)
                                printable_y = 15
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
        # Destroy opencv windows
        cap.release()
        cv2.destroyAllWindows()
    # Input: Directory 
    if dirc:
        framesCount=0
        for image in images:
            imgAux=cv2.imread(image, 1)
            
            if descriptorType == 1:
                faces=faceDescriptor.detectMultiScale(imgAux, 1.1, 4)
                index=0
                for (x, y, w, h) in faces:
                    crop_img=cv2.resize(imgAux[y:y+h, x:x+w], (224, 224))
                    expanded=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(expanded)

                    plotEmotions(pred, str(framesCount) + \
                                    "_"+str(index), crop_img)
                    index=index + 1
                framesCount=framesCount + 1
            # DLib face descriptor
            elif descriptorType == 0:
                dets=faceDescriptor(imgAux, 0)
                index=0
                for i, d in enumerate(dets):
                    crop_img=cv2.resize(imgAux[d.top():d.top(
                    )+d.bottom()-d.top(), d.left():d.left()+d.right()-d.left()], (224, 224))
                    expanded=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(expanded)

                    plotEmotions(pred, str(framesCount) + \
                                    "_"+str(index), crop_img)
                    index=index + 1
            # DNN OpenCV face descriptor
            else:
                (h, w) = imgAux.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(imgAux, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

                model.setInput(blob)
                detections = model.forward()
                index=0
                for i in range(0, detections.shape[2]):
                    if (detections[0, 0, i, 2] > 0.6):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        crop_img=cv2.resize(imgAux[startY:endY, startX:endX], (224, 224))
                        expanded = tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(expanded)

                        plotEmotions(pred, str(framesCount)+'_'+str(index), crop_img)
                        index=index + 1
                framesCount=framesCount + 1
    # Input: Image
    else:
        try:
            # OpenCV face descriptor
            if descriptorType == 1:
                faces = faceDescriptor.detectMultiScale(imageSolo, 1.1, 4)
                index=0
                for (x, y, w, h) in faces:
                    crop_img=cv2.resize(
                        imageSolo[y:y+h, x:x+w], (224, 224))
                    imgAux=tf.expand_dims(crop_img, axis=0)
                    pred=model_builded.predict(imgAux)

                    plotEmotions(pred, str(index), crop_img)
                    index=index + 1
            # DLib face descriptor
            elif descriptorType == 0:
                dets=faceDescriptor(imageSolo, 0)
                if len(dets) is not 0:
                    index=0
                    for i, d in enumerate(dets):
                        crop_img=cv2.resize(
                                frameSaved[d.top():d.top()+
                                d.bottom()-d.top(),
                                d.left():d.left()+
                                d.right()-d.left()
                            ], (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(imgAux)

                        plotEmotions(pred, str(index), crop_img)
                        index=index + 1
            # DNN OpenCV face descriptor
            else:
                (h, w) = imageSolo.shape[:2]
                blob = cv2.dnn.blobFromImage(cv2.resize(imageSolo, (224, 224)), 1.0, (224, 224), (104.0, 177.0, 123.0))

                model.setInput(blob)
                detections = model.forward()
                index=0
                for i in range(0, detections.shape[2]):
                    if (detections[0, 0, i, 2] > 0.6):
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")
                        crop_img=cv2.resize(imageSolo [startY:endY, startX:endX], (224, 224))
                        imgAux=tf.expand_dims(crop_img, axis=0)
                        pred=model_builded.predict(imgAux)

                        plotEmotions(pred, str(index), crop_img)
                        index=index + 1
        # Handling error when escaping camera
        except NameError:
            sys.exit(0)
