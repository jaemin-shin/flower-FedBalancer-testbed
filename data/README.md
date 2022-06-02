# UCI-HAR Dataset

We preprocess the Human Activity Recognition data released by [Davide Anguita et al.](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones).
The dataset contains accelerometer and gyroscope data from 30 volunteers performing six activities (WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING).

## Setup Instructions

1. Install required packages
```
$ pip install pandas
```

2. Create subdirectory named ```data``` in this directory, and create sub-subdirectory names ```train``` and ```test```. Then, change directory.
```
$ mkdir -p data
$ mkdir -p data/train
$ mkdir -p data/test
$ mkdir -p har/train/0
$ mkdir -p har/train/1
$ mkdir -p har/train/2
$ mkdir -p har/train/3
$ mkdir -p har/train/4
$ mkdir -p har/train/5
$ cd data
```
3. Download ```UCI HAR Dataset.zip``` file [here](https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) to the ```data``` subdirectory.
4. Unzip the data at a ```data``` subdirectory with the following command, and change to the parent directory.
```
$ unzip 'UCI HAR Dataset.zip'
$ cd ..
```
5. Run preprocessing.py to generate .json files in ```data/train``` and ```data/test``` and asset files for Android training.
```
$ python preprocessing.py
```
