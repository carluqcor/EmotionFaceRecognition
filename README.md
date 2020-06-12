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
| Parameter   | Description | Command       |
|    :----:   |    :----:   |    :----:     |
| Header      | Title       | Here's this   |
| Paragraph   | Text        | And more      |


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
- [TensorFlow](https://flutter.dev/) - Beautiful native apps in record time.
- [Keras](https://developer.android.com/studio/index.html/) - Tools for building apps on every type of Android device.

## Author
- **Carlos Luque Córdoba** - lead developer: [GitHub](https://github.com/i62lucoc) & [LinkedIn](www.linkedin.com/in/carlosluquecordoba).


## License
This project is licensed under the GNU GPL v3 License - see the [LICENSE.md](LICENSE.md) file for details.
