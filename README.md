# Machine Learning Visualization Tool

## About / Synopsis

* It is simple project for visualization binary CNN classification

## Table of contents

> * [Machine Learning Visualization Tool](#title--repository-name)
>   * [About / Synopsis](#about)
>   * [Table of contents](#table-of-contents)
>   * [Installation](#installation)
>   * [Usage](#usage)
>   * [Features](#features)
>   * [Screenshots](#screenshots)

## Installation

* Server requires python3.6+
* Install `pipenv`
```
pip install pipenv
```
* Go to project main directory
* Install all dependencies via `Pipfile.lock`
```
pipenv install --ignore-pipfile
```
* pipenv will automatically create virtualenv for your project

In order to run the application you have to create configuration file (`config.yml`) and save it under `config` directory.
* It contains paths where your dataset images are stored
    * `raw` paths are the paths on the system from where server should copy images to `transformed` paths
## Usage
* go to project main directory
* set PYTHONPATH
```
# for Linux users
export PYTHONPATH=.

# for Windows (Powershell) users
$env:PYTHONPATH = "."
```
* start the server
```
python ./mlvt/server/run.py
```

* in your web browser go to `localhost:5000/home`
* you can also check the API via Swagger UI at `localhost:5000/ui`

### Features
* Provides convinient way to label your unlabelled images
* Observe your model training process with live updating accuracy and loss plots
* You can switch between datasets without restarting the server
* Test your image - you can test model against your own image

### Screenshots
![Alt text](http://https://github.com/Infam852/MLVT/images/Predictions.jpg "Set labeles to unlabelled images")

