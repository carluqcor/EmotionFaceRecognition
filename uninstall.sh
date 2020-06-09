#!/bin/bash
# Remove pipenv environment
pipenv --rm

# Remove project
cd ../
rm -rf EmotionFaceRecognition/

echo -e 'Project removed succesfully!'
