# Credit Score Classification Project

![Credit Score Classification](dataset-cover.jpg)

## Overview

This repository contains a credit score classification project that aims to predict creditworthiness based on historical data. The project utilizes machine learning algorithms to analyze various features related to individuals' financial behavior and determine their credit score bucket. 

The credit score of a person determines the creditworthiness of the person. It helps financial companies determine if you can repay the loan or credit you are applying for.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Dataset

The dataset used for this project can be found on [Kaggle](https://statso.io/credit-score-classification-case-study/) and it is also stored in the `data` directory. It includes anonymized information on various customers, such as income, age, credit history, loan amount, and their corresponding credit score class. The data is split into a training set and a test set, located in the `data/train.csv` and `data/test.csv` respectively.

## Installation

To use this project locally, you'll need to follow these steps:

1. Clone this repository to your local machine.
2. Python version `Python = 3.10.6`
3. Install the required dependencies by running `pip install -r requirements.txt`.

## Usage

1. Ensure you have installed all the dependencies as mentioned in the Installation section.
2. Use Jupyter Notebook or any other Python environment to run the provided scripts.
3. Explore the provided Jupyter notebooks in the `notebooks` directory to understand the project workflow and analysis.

## Features

The project offers the following features:

1. Data Preprocessing: Clean and preprocess the dataset to handle missing values and prepare it for machine learning models.
2. Feature Engineering: Extract relevant features and perform necessary transformations to enhance model performance.
3. Model Selection: Experiment with different classification algorithms like Random Forest, Logistic Regression, etc.
4. Model Evaluation: Evaluate models using various metrics like accuracy, precision, recall, F1-score, and ROC-AUC to assess performance.
5. Hyperparameter Tuning: Optimize model performance by tuning hyperparameters using techniques like Grid Search.
6. Model Deployment: Deploy the best performing model for real-world predictions.

## Model Training

To train the credit score classification model:

1. Run the script `train.py` to train the models using the training data.
2. The script will automatically save the trained models in the `models` directory.

## Evaluation

To evaluate the trained models:

1. Run the script `evaluate.py` using the test data.


## Results

The final model's performance and evaluation metrics will be available in the `results` directory. Additionally, the Jupyter notebooks in the `notebooks` directory provide a detailed analysis of the project.

## Contributing

Contributions to this project are welcome! If you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request. Please follow the standard GitHub workflow when contributing.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

If you have any questions or need further assistance, please feel free to contact the project owner:

- Name: Saurabh Nirwan
- Email: saurabhnirwan11@gmail.com
- LinkedIn: [Saurabh Nirwan](https://www.linkedin.com/in/saurabh-nirwan-468a9683)
- Kaggle:  [Saurabh Nirwan](https://www.kaggle.com/saurabhnirwan)