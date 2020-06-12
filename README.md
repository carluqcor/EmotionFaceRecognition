# EmotionFaceRecognition
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

## Demo
```bash
# Access to demo executable
$ cd Executables/

# Execute demo with default values (camera, AVG, DNN and windowSize = 3)
$ python3 demo.py -r folder/

# Watch out! If you are going to execute demo more than once give other folder pathname or delete the existing one
$ python3 demo.py -r folderDNN/ -i ../Images/ -m ../Models/vgg19.h5 -d DNN
```
<a href="https://github.com/i62lucoc/EmotionFaceRecognition/stargazers">
  <img src="https://img.shields.io/github/stars/i62lucoc/EmotionFaceRecognition.svg?style=for-the-badge">
</a>


## Uninstall
```bash
# Give permissions to execute uninstall script
$ chmod 777 uninstall.sh

$ ./uninstall.sh
´´´

## Quick start
Escribe aquí los pasos básicos para ejecutar la demo.

```bash
cd XXX
python xxx/demo.py --args
```

Inserta aquí una figura de la demo en acción.
