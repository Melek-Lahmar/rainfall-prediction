# LinkedIn Post

I completed an end-to-end Australian rainfall prediction project as part of IBM's *Machine Learning with Python* coursework and expanded it into a reproducible portfolio repository.

The objective was to classify rainfall for the Melbourne area using historical weather observations while paying close attention to target definition and data leakage.

I implemented:

- data cleaning and season feature engineering;
- leakage-aware `ColumnTransformer` and scikit-learn pipelines;
- Logistic Regression and Random Forest baselines;
- stratified cross-validation and `GridSearchCV`;
- evaluation with accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves;
- model interpretation and serialization of the complete fitted pipeline.

The selected **Random Forest (tuned)** achieved **0.841 accuracy**, **0.752 precision**, **0.492 recall**, **0.595 F1**, and **0.885 ROC-AUC** on the held-out test set.

The biggest lesson was that a strong classification project is not just about accuracy: defining the prediction time correctly, preventing leakage, and evaluating the minority class are equally important.

Technologies: Python, Pandas, NumPy, Matplotlib, scikit-learn, Jupyter, and joblib.

GitHub: https://github.com/Melek-Lahmar/australian-rainfall-prediction

#MachineLearning #Python #DataScience #ScikitLearn #IBM #ArtificialIntelligence

## LinkedIn Projects Section Description

Built and validated a leakage-aware rainfall classifier for the Melbourne area using scikit-learn pipelines, stratified cross-validation, and GridSearchCV. Compared Logistic Regression and Random Forest across accuracy, precision, recall, F1, and ROC-AUC, then saved the complete final pipeline for reproducible inference.

## Credential

[Add IBM/Coursera credential URL after course completion]
