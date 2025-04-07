# Letter Recognition
This project is an experimental project on letter recognition.

## Install Dependencies
The dependencies of this project are managed using Conda.
### Install Conda on ARM64 (Apple Silicon)
```shell
$ curl -o Miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
$ shasum -a 256 Miniconda.sh
$ bash Miniconda.sh
$ conda --version
$ conda init zsh
$ exec $SHELL
```
### Init Conda Environment
```shell
$ conda create -n letter_recognition python=3.10
$ conda activate letter_recognition
```
### Install Dependencies
```shell
$ conda install -c conda-forge \
      numpy pandas scipy matplotlib \
      opencv pdf2image pytesseract spacy
$ pip install spacy_llm
$ conda install -c apple tensorflow-deps
$ pip install tensorflow-macos tensorflow-metal
$ conda install -c conda-forge keras tensorflow-datasets
$ conda install -c hyperopt
```
### Export Conda Environment
```shell
$ conda env export > environment.yml
```
### Import Conda Environment
```shell
$ conda env create -f environment.yml
```

## How to Run
### Find Optimal CNN
Uncomment `hyperparameter optimization step` code block in `main.py`
```shell
$ python main.py
```
### Test CNN
Uncomment `model testing step` code block in `main.py`
```shell
$ python main.py
```
### Run Program
Uncomment `real program` code block in `main.py`
```shell
$ python main.py
```
### Convert PDF to Text and Find Entities
Uncomment `find entities from pdf` code block in `main.py`
```shell
$ python main.py
```

## TODO
- CNN model
  - Test by using real camera but, the input needs to be sampled to ease the test observation

## Question
- NLP model
  - How to find the relation between entities?
