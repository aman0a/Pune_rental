DATASET LINK : https://www.kaggle.com/datasets/epigos/ghana-house-rental-dataset

Pune Rental Price Prediction using Machine Learning
Overview
This project focuses on predicting rental prices for residential properties in Pune, India, using machine learning algorithms. The dataset contains 22,801 rental listings with various features like property type, location, number of bedrooms, area, and furnishing type. After training several machine learning models, Random Forest emerged as the most accurate model for predicting rental prices, outperforming other models such as XGBoost, CatBoost, Support Vector Regression (SVR), and Linear Regression.

Dataset
The dataset used in this project contains 22,801 rental listings with the following features:

seller_type: Type of seller (e.g., OWNER)

bedroom: Number of bedrooms in the property

layout_type: Type of layout (e.g., BHK, RK)

property_type: Type of property (e.g., Independent Floor, Apartment)

locality: Area/locality where the property is located

price: Rental price of the property (target variable)

area: Total area of the property (in square feet)

furnish_type: Furnishing type (e.g., Furnished, Semi-Furnished, Unfurnished)

bathroom: Number of bathrooms in the property

The dataset can be accessed from Pune Rent Dataset.

Libraries Used
pandas: For data manipulation and analysis

numpy: For numerical computations

scikit-learn: For implementing machine learning algorithms

matplotlib: For data visualization

seaborn: For advanced visualization

catboost: For the CatBoost model

xgboost: For the XGBoost model

To install the necessary libraries, run:

bash
Copy
pip install pandas numpy scikit-learn matplotlib seaborn catboost xgboost
Model Overview
Models Trained:
Random Forest: Best-performing model with an R² value of 7.7.

XGBoost: Strong performance but slightly behind Random Forest in terms of accuracy.

CatBoost: Performed well, but Random Forest provided better accuracy.

SVR: Showed poorer performance in terms of RMSE and R².

Linear Regression: Baseline model for comparison.

Evaluation Metrics:
R² Score: Measures the proportion of variance explained by the model. Higher values indicate better predictions.

Root Mean Squared Error (RMSE): Measures the average magnitude of the error.

Mean Absolute Error (MAE): Indicates the average absolute error between predicted and actual values.

Best Performing Model: Random Forest
R²: 7.7

RMSE: 2,360.7 (example)

MAE: 1,872.5 (example)

Methodology
Data Collection and Preprocessing

The dataset was collected from real estate websites.

Data cleaning was performed, including handling missing values, removing outliers, and encoding categorical features.

Feature scaling was applied using StandardScaler for models like SVR.

Feature Engineering

Key features like location, number of bedrooms, and furnishing type were selected for model training.

Geospatial data (latitude and longitude) was processed for location-based predictions.

Model Selection and Hyperparameter Tuning

Hyperparameters for each model were tuned using GridSearchCV and RandomizedSearchCV.

Models were evaluated using k-fold cross-validation to improve generalization.

Results
Random Forest outperformed all other models in terms of prediction accuracy, explaining the largest portion of variance in rental prices.

XGBoost performed well but couldn't match the precision of Random Forest.

SVR and Linear Regression had higher error rates, making them less suitable for this problem.

Conclusion
This project demonstrates the power of machine learning in predicting rental prices for properties in Pune. Random Forest proved to be the most accurate model for this dataset, providing better decision-making tools for tenants, landlords, and real estate investors.

Future Work
Integrate time-series data to track rental price trends over time.

Incorporate additional features like proximity to transportation, education institutions, etc.

Build a real-time prediction tool for ongoing rental listings.

Files
Pune_rent.csv: The dataset used for training the models.

ML_FINAL (1).ipynb: The Jupyter notebook containing the complete implementation of the project.

RP ML.pdf: The research paper providing a detailed analysis of the project.

