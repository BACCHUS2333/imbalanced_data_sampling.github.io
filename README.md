# imbalanced_data_sampling.github.io

**Resampling Methods Comparison for Fraud Detection**

## Overview

This repository explores the effectiveness of three resampling methods—undersampling, oversampling, and SMOTE (Synthetic Minority Over-sampling Technique)—in the context of fraud detection using transaction data. We evaluate these methods with three different machine learning algorithms: Lasso Regression, Decision Tree, and Logistic Regression. The goal is to determine which resampling method performs best for each algorithm in identifying fraudulent transactions.

## Dataset

The dataset used for this analysis contains transaction data, including features such as transaction amount, merchant ID, and customer information. The target variable is a binary indicator of whether a transaction is fraudulent or not.

## Resampling Methods

### 1. Undersampling
Undersampling involves reducing the number of instances in the majority class (non-fraudulent transactions) to balance the class distribution with the minority class (fraudulent transactions).

### 2. Oversampling
Oversampling increases the number of instances in the minority class by replicating or generating synthetic samples, aiming to balance the class distribution.

### 3. SMOTE (Synthetic Minority Over-sampling Technique)
SMOTE generates synthetic samples for the minority class by interpolating between existing minority class instances, thereby addressing the class imbalance issue.

## Machine Learning Algorithms

We apply three popular machine learning algorithms to the resampled datasets:

### 1. Lasso Regression
Lasso Regression is a linear regression technique that incorporates a penalty term (L1 regularization) to shrink coefficients, effectively performing variable selection and regularization.

### 2. Decision Tree
Decision Tree is a non-linear algorithm that partitions the feature space into regions, making predictions based on the majority class within each region.

### 3. Logistic Regression
Logistic Regression is a linear model used for binary classification, estimating the probability of an instance belonging to a particular class.

## Methodology

For each combination of resampling method and machine learning algorithm, we train the model on the resampled training data and evaluate its performance using appropriate metrics such as accuracy, precision, recall, and F1-score. We repeat this process for multiple iterations to account for variability in results.

## Results and Discussion

The results section will summarize the performance of each combination of resampling method and machine learning algorithm. We'll discuss the strengths and weaknesses of each approach and provide insights into which combination is most effective for fraud detection in our dataset.

## Conclusion

In conclusion, we'll summarize our findings and recommendations for selecting the optimal resampling method and machine learning algorithm for fraud detection tasks based on the characteristics of the dataset.

## Repository Structure

```
|- data/
   |- transaction_data.csv   # Raw transaction data
|- notebooks/
   |- data_exploration.ipynb   # Jupyter notebook for data exploration
   |- model_training.ipynb      # Jupyter notebook for model training and evaluation
|- README.md                    # This README file
```

## Usage

To reproduce the analysis:

1. Clone this repository to your local machine.
2. Navigate to the notebooks directory.
3. Open and run the Jupyter notebooks in the following order: `data_exploration.ipynb`, `model_training.ipynb`.

## Dependencies

- Python 3.11
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Contributors

- [Your Name]
- [Contributor 2]
- [Contributor 3]

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

> [!NOTE]
> the data is provided by tutor
