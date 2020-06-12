<h1 align="center">EmotionFaceRecognition</h1>

<p align="center">
  <a href="https://www.gnu.org/licenses/gpl-3.0.en.html">
    <img src="https://img.shields.io/github/license/i62lucoc/EmotionFaceRecognition.svg?style=for-the-badge">
  </a>
  <a href="https://github.com/i62lucoc/EmotionFaceRecognition/stargazers">
    <img src="https://img.shields.io/github/stars/i62lucoc/EmotionFaceRecognition.svg?style=for-the-badge">
  </a>
</p>

## Description
EmotionFaceRecognition is  

## Table of Contents
* [Instalation](#instalation)
* [Demo parameters](#demo-parameters)
* [Demo](#demo)
* [Uninstall](#uninstall)
* [Built with](#built-with)
* [License](#license)

## Instalation
```bash
# Repository files
$ git clone git@github.com:i62lucoc/EmotionFaceRecognition.git

# Download large files from Google Drive
$ python3 EmotionFaceRecognition/download.py

# Install pipenv (Recommended)  or install libraries manually
$ pip3 install pipenv
$ cd EmotionFaceRecognition
$ pipenv install

# Activate shell for executables
$pipenv shell
```
## Demo parameters
| Parameter   | Description | Command       | Required       | Default       | Options |
|    :----:   |    :----:   |    :----:     |    :----:     |    :----:     |    :----:     | 
| Results Directory      | Directory path to save results    | -r   |True   |    None      | None |
| Confidence   | Confidence to accept a face detection from DNN OpenCV detector   | -c | False | 0.75 | None |
| Model File   | Trained model filename   | -m | False  | vgg19.h5 | None |
| Window Size   | Integer number to apply temporal window operations  | -w | False | 3 | < 2 (Deactivate) \ > 1 (Activate) |
| Operation   | Operation to apply when temporal window is activated   | -o | AVG | AVG \ MEDIAN \ MAX  |
| Face Detector  | Select which face detector is going to be used | -d | False | 'DNN' | DNN \ OPENCV \ DLIB |
| Media Input  |  Select input media  | -i | False | '0' | 0 (camera) \ video file | image file | folder with images |


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
$ python3 demo.py -r folderOPENCV/ -i ../Videos/se01.mp4 -m ../Models/vgg19.h5 -d OPENCV
```

### Visualization emotion predict frame from video 
![alt text](https://github.com/i62lucoc/EmotionFaceRecognition/blob/master/Screenshots/VideoHappy.png?raw=true)

## Uninstall
```bash
# Give permissions to execute uninstall script
$ chmod 777 uninstall.sh

$ ./uninstall.sh
```

## Built with
- [TensorFlow](https://github.com/tensorflow/tensorflow) - Tools for machine learning.
- [Keras](https://github.com/keras-team/keras) - Tools neural networks.

## Author
- **Carlos Luque Córdoba** - lead developer: [GitHub](https://github.com/i62lucoc) & [LinkedIn](www.linkedin.com/in/carlosluquecordoba).


## License
This project is licensed under the GNU GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details.
