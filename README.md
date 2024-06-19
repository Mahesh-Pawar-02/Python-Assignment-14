# Play Predictor using K-Nearest Neighbors (KNN)

This project implements a K-Nearest Neighbors (KNN) classifier to predict whether to play based on weather and temperature conditions. The data is encoded using `LabelEncoder`, and the model's accuracy is evaluated with different values of `k`.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Code Overview](#code-overview)
- [Data Preparation](#data-preparation)
- [Model Training and Prediction](#model-training-and-prediction)
- [Accuracy Evaluation](#accuracy-evaluation)
- [License](#license)

## Installation

1. Clone the repository:
   ```
    git clone https://github.com/your-username/play-predictor.git
    cd play-predictor
    ```

3. Install the required packages:
    ```
    pip install pandas scikit-learn
    ```

4. Ensure you have the data file `PlayPredictor.csv` in the project directory.

## Usage

Run the `m.py` script to train the model, make a prediction for a test case, and evaluate the model's accuracy:
```
python CaseStudy.py
