# aa228-final-project

This is the final project for Stanford aa228 Fall 2019.

Examine the write up in `final.pdf` for an explanation of the project.

To train the parameters of the Kalman filter, run:

`pipenv run python local_search.py <file_name> <increment>`

The `file_name` parameter is the name of a text file where progress of the learning algorithm will get saved. If you start the script again it will read the last result in the file specified and start training from there.

The `increment` parameter tells the gradient ascent algorithm how large of a step to try for any of the parameters in any direction.

You may need to run this command first:

`pipenv install`

If this fails, make sure you have pipenv installed.

