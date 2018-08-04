# Author Identification Project

Kahlil Khan /
kkhan37 /
CS-6460 Educational Technology /
Summer 2018

## Description of files
* **open.ipynb**:  Jupyter notebook.  Trains the author identification model.  Status: Complete.
* **origin.py**:  Python.  The external tool.  Provides interface to connect Canvas.  Status:  Draft interface works command line, not yet with Canvas (requires JSON formatting)
* **origin-test.ipynb**:  Jupyter notebook.  Demonstration surogate for Canvas LMS.
* **lti.ipynb**:  Jupyter notebook.  Alpha-level attempt to connect to Canvas via its LTI version 1 and 2 interfaces.
* **environment.yml:**  Text.  Listing of development dependencies.  See installation instructions below.

## Description of data
* **data/C50/C50all/**:  5051 articles from 50 authors.  This is a combination of the training and test data from Reuter_50_50 dataset
* **data/5-300-lstm128-128-20-50-model.h5**:  The trained Keras model.  Trained by open.ipynb and used in orgin.py.

## Installation instructions
1. Install and run mongo database: [macos](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-os-x/) or [linux](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/).
1. Install anaconda from https://www.anaconda.com
1. ```conda env create -f environment.yml```
1. ```source activate tensorflow_p36```
1. ```jupyter lab```
1. Open **origin-test.ipynb**
