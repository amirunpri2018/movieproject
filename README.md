# Film Success Project

This project uses supervised machine learning techniques to learn features significant to movie success, and predicts a movie's revenue and ratings based on a series of input responses from the user. We use features from the IMDB movie dataset, and hand-craft modifications of these variables to serve as more scaled and predictive features. We experiment with linear regression, ridge, and lasso methods.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See running the project for notes on how to deploy the project on a live system.

### Prerequisites

You will need to have the following packages installed.
* csv
* matplotlib
* numpy
* sklearn
* collections

## Running the project

To run the python file, execute 
```
python predict.py
``` 
with the dataset, data.csv in the same folder. The file is broken down into four main sections: 
* "Global Constants and Variables" at the top
* "Load Data" for importing the csv file and selecting the desired predictors from it with feature extraction
* "Data / Feature Modification" for assisting in other attempts in feature extraction
* "Modeling / Prediction" for running regression techniques from sklearn, printing train error, validation error, and test error statistics, and taking user queries for film rating and revenue prediction
* "Create Bayesian Network" for printing statistics on the probability of certain features, and conditioning on these features for the probability of being in a certain bucket of range for rating and revenue. 

At the bottom is a main() function which initializes the workload of the file. Once the file is run with the execution described, several prompts will be given to the user about queries they are interested in, including a series of questions to design a movie of your own and have the option to predict its revenue or rating using our developed models. This query can be repeated as many times as desired or exited to complete execution of the program. 

## Authors

* **Andrew Narcomey**, Stanford University, Department of Computer Science
* **Austin Narcomey**, Stanford University, Department of Computer Science
* **Jonathan Zhu**, Stanford University, Department of Computer Science
