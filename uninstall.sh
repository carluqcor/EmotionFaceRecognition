#!/bin/bash
# If not in VGG19demo-evaluator/ do
cd ./i62lucoc/

# Remove pipenv environment
pipenv --rm

# Remove project
cd ../../
rm -rf VGG19demo-evaluator/

# Self destroy this bash file
echo -e 'Project removed succesfully!'
