<h1 align="center">EmotionFaceRecognition</h1>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">
    <img src="https://img.shields.io/github/license/i62lucoc/EmotionFaceRecognition.svg?style=for-the-badge">
  </a>
  <a href="https://github.com/i62lucoc/EmotionFaceRecognition/stargazers">
    <img src="https://img.shields.io/github/stars/i62lucoc/EmotionFaceRecognition.svg?style=for-the-badge">
  </a>
</p>

### Description
EmotionFaceRecognition is a software designed to predict emotions from facial expressions written in Python. The purpose of this project is to allow a machine to be capable to detect face expressions while predicting the emotions.

## Table of Contents
* [Install](#install)
* [Demo parameters](#demo-parameters)
* [Demo](#demo)
* [Uninstall](#uninstall)
* [Built with](#built-with)
* [Credits](#credits)
* [Author](#author)
* [License](#license)

## Install
```bash
# Repository files
$ git clone git@github.com:i62lucoc/EmotionFaceRecognition.git

# Download large files from Google Drive
$ cd EmotionFaceRecognition/
$ python3 download.py

# Install pipenv (Recommended)  or install libraries manually
$ pip3 install pipenv
$ pipenv install

# Activate shell for executables
$ pipenv shell
```
## Demo parameters
| Parameter   | Description | Command       | Required       | Default       | Options |
|    :----:   |    :----:   |    :----:     |    :----:     |    :----:     |    :----:     | 
| Results Directory      | Directory path to save results    | -r   |True   |    None      | None |
| Confidence   | Confidence to accept a face detection from DNN OpenCV detector   | -c | False | 0.75 | None |
| Model File   | Trained model filename   | -m | False  | vgg19.h5 | None |
| Window Size   | Integer number to apply temporal window operations  | -w | False | 3 | < 2 (Deactivate) <br/> > 1 (Activate) |
| Operation   | Operation to apply when temporal window is activated   | -o | False | AVG | AVG <br/> MEDIAN <br/> MAX  |
| Face Detector  | Select which face detector is going to be used | -d | False | DNN | DNN <br/> OPENCV <br/> DLIB |
| Media Input  |  Select input media  | -i | False | 0 | 0 (Camera) <br/> Video file <br/> Image file <br/> Folder with images |


## Demo
```bash
# Access to demo executable
$ cd Executables/

# Execute demo with default values (camera, AVG, DNN OpenCV detector and windowSize = 3)
$ python3 demo.py -r folder/

# Watch out! If you are going to execute demo more than once give other folder pathname or delete the existing one
$ python3 demo.py -r folderDNN/ -i ../Images/ -m ../Models/vgg19.h5 -d DNN
```
### Output in folderDNN
![alt text](https://github.com/i62lucoc/EmotionFaceRecognition/blob/master/Screenshots/FolderOutput.png?raw=true)

### Visualization emotion predict 
![alt text](https://github.com/i62lucoc/EmotionFaceRecognition/blob/master/Screenshots/FolderOutputPic.png?raw=true)


```bash
# Execute demo with a video and using OpenCV HaarCascade detector
$ python3 demo.py -r folderDNN/ -i ../Videos/se01.mp4 -m ../Models/vgg19.h5 -d DNN
$ python3 demo.py -r folderOPENCV/ -i ../Videos/se01.mp4 -m ../Models/vgg19.h5 -d OPENCV
$ python3 demo.py -r folderDLIB/ -i ../Videos/se01.mp4 -m ../Models/vgg19.h5 -d DLIB
```

### Visualization of emotions predicted on video with different facial detectors
#### DNN OpenCV
[![Emotions](https://i.imgur.com/DBPibc9.png)](https://youtu.be/_ACrY9N-GKA "Emotions")

#### HaarCascade OpenCV
[![Emotions](https://i.imgur.com/7iMDj1P.png)](https://youtu.be/43dyeEAV6Gg "Emotions")

#### DLIB
[![Emotions](https://i.imgur.com/ycEVy4L.png)](https://youtu.be/Xbz5LZg84FA "Emotions")

## Uninstall
```bash
# Give permissions to execute uninstall script
$ chmod 777 uninstall.sh

$ ./uninstall.sh
```

## Built with
- [TensorFlow](https://github.com/tensorflow/tensorflow) - Tools for machine learning.
- [Keras](https://github.com/keras-team/keras) - Tools for convolutional neural networks.
- [OpenCV](https://github.com/opencv/opencv) - Tools for image processing
- [EmotioNet](http://cbcsl.ece.ohio-state.edu/EmotionNetChallenge/) - Database for training and validation.
- [EmotionFACS](https://imotions.com/blog/facial-action-coding-system/) - Database for training and validation.
- [AffectNet](http://mohammadmahoor.com/affectnet/) - Database for training and validation.

## Credits
- [Video for evaluating model and demo](https://www.youtube.com/watch?v=y2YUMPJATmg)

## Authors
- **Carlos Luque Córdoba** - lead developer: [GitHub](https://github.com/i62lucoc) & [LinkedIn](www.linkedin.com/in/carlosluquecordoba).
- **Manuel J. Marín-Jiménez** - advisor: [GitHub](https://github.com/mjmarin)


## License
This project is licensed under the GNU GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details.
