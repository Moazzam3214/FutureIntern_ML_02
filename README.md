# IRIS Dataset Classification

This project involves building a machine learning classification model to predict the species of iris flowers based on four features: sepal length, sepal width, petal length, and petal width. The classic **IRIS dataset** is widely used in data science and machine learning for classification tasks.

## Project Overview

The goal of this project is to train and evaluate classification models using the IRIS dataset and compare their performance in terms of accuracy, precision, recall, and other relevant metrics.

### Dataset

The dataset consists of 150 samples, each with the following features:
- **Sepal Length** (in cm)
- **Sepal Width** (in cm)
- **Petal Length** (in cm)
- **Petal Width** (in cm)

There are three target classes representing the species of iris flowers:
- **Setosa**
- **Versicolor**
- **Virginica**


## Technologies Used

- Python
- Jupyter Notebook
- Pandas and NumPy for data processing
- Matplotlib and Seaborn for visualization
- Scikit-Learn for machine learning

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/moazzam3214/iris-dataset-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd iris-dataset-classification
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook IRIS_dataset_classification.ipynb
   ```
2. Run the notebook cells to load the dataset, train the models, and make predictions.

## Models

The following classification models are used in this project:
- **Logistic Regression**
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**

The models are evaluated based on their classification performance using metrics like:
- **Accuracy**
- **Confusion Matrix**
- **Precision, Recall, F1-Score**

## Results

The models are compared based on their prediction accuracy, confusion matrix, and precision-recall values.

## Future Improvements

- Experiment with more advanced classifiers like **XGBoost** or **Neural Networks**.
- Perform hyperparameter tuning to further optimize the model's performance.
- Deploy the model using Flask or Streamlit for easy user interaction.

## Contributing

Contributions are welcome! Feel free to fork this repository, make changes, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
