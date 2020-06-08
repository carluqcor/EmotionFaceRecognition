#! /bin/bash
# If not in VGG19demo-evaluator/ do
cd ~/VGG19demo-evaluator/

# Remove pipenv environment
pipenv --rm

# Remove project
cd ..
rm -r VGG19demo-evaluator/

# Self destroy this bash file
echo -e 'Project removed succesfully!'
rm -- "$0"
