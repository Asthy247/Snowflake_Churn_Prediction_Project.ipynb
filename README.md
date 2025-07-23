# Customer Churn Prediction Data Pipeline on Snowflake & Google Colab

**Project Overview**
This project demonstrates an end-to-end data pipeline for customer churn prediction, leveraging Snowflake for robust data warehousing and feature engineering, and Google Colaboratory (Colab) with Python for machine learning model training, evaluation, and advanced visualization. The goal is to identify customers at risk of churning, providing actionable insights for retention strategies.


**Key Technologies Used:**

**Snowflake:** Cloud Data Warehouse (Data Ingestion, ELT, Feature Engineering, Analytical Queries)

**Google Colaboratory (Colab):** Cloud-based Jupyter Notebook environment for Python

**Python Libraries:** pandas, scikit-learn, matplotlib, seaborn, snowflake-connector-python


# 1. Data Pipeline on Snowflake
The data pipeline is structured into distinct layers within Snowflake to ensure data quality, scalability, and maintainability.


**1.1. Setup:** Database, Schemas, and Warehouse
A dedicated database (CHURN_PREDICTION_DW) was created, along with schemas for different stages of the data pipeline:

**RAW:** For initial, unprocessed data.


**STAGING:** For cleaned and type-casted data.


**FEATURES:** For engineered features ready for machine learning.


**ANALYTICS:** For storing model predictions and aggregated analytical results.



**1.2. Data Ingestion (Simulated RAW Layer)**
In a real-world scenario, data would be ingested from various sources (e.g., S3, Azure Blob) using Snowflake's COPY INTO command. 
For this project, the RAW layer is simulated by directly inserting sample customer data into RAW.CUSTOMER_DATA_RAW using VALUES clauses. 
This table holds raw, untransformed data, including the target CHURN_LABEL.



**1.3. Staging Layer (ELT - Initial Cleaning & Type Casting)**
The STAGING.CUSTOMER_DATA_STG table performs initial transformations:



**Data Type Conversion:** Raw string columns (TENURE_MONTHS, MONTHLY_CHARGES, TOTAL_CHARGES) are converted to appropriate numerical types (INTEGER, NUMERIC). Boolean-like strings ('Yes'/'No') are converted to actual BOOLEAN types.


**Basic Data Cleaning:** Handles missing or blank TOTAL_CHARGES (common for new customers) by imputing them to 0 or MONTHLY_CHARGES.


**1.4. Feature Engineering Layer (Preparing Data for ML)**
The FEATURES.CUSTOMER_CHURN_FEATURES table is the core input for the machine learning model. This layer focuses on creating and preparing features:


**New Feature Creation**: AVG_CHARGE_PER_MONTH is calculated from TOTAL_CHARGES and TENURE_MONTHS.


**Categorical Encoding:** Categorical features like GENDER, INTERNET_SERVICE, CONTRACT_TYPE, and PAYMENT_METHOD are one-hot encoded directly in SQL using IFF statements, 
transforming them into numerical (boolean) columns suitable for machine learning algorithms.


**Data Imputation:** Further handling of TOTAL_CHARGES for new customers to ensure numerical integrity.


**1.5. Model Integration (Simulated Prediction Storage)**
To simulate the end-to-end workflow, a table ANALYTICS.CUSTOMER_CHURN_PREDICTIONS is created to store hypothetical churn probabilities and predicted labels 
from an external ML model. In a production environment, this table would be populated by the actual model's output.


**1.6. Analytical Queries & Insights**
Several SQL queries are run against the FEATURES and ANALYTICS layers to derive business insights and evaluate the (simulated) model performance:


**Overall Churn Rate (Actual & Predicted):** Provides a high-level view of churn.


**Model Accuracy**: A basic measure of how well the simulated predictions align with actual churn.


**High-Risk Customers:** Identifies customers predicted to churn but who haven't yet, highlighting intervention opportunities.


**Churn Rate by Contract Type:**


**Insight:** Customers on 'Month-to-month' contracts show a significantly higher churn rate compared to 'One year' or 'Two year' contracts in the simulated data, a common real-world pattern.


**Churn Rate by Payment Method:**


**Insight:** Reveals if certain payment methods correlate with higher churn, guiding targeted retention efforts.


**Distribution of Numerical Features (Tenure, Monthly Charges):**


**Insight:** Provides descriptive statistics (min, max, quartiles, average) to understand customer lifecycle and spending habits.


**False Positives & False Negatives Analysis:**


**Insight:** Crucial for understanding model errors; False Positives are potential retention targets, False Negatives are missed opportunities.


**High-Value Customers**: Identifies customers with long tenure and high total charges, who are critical for retention focus.



# 2. Machine Learning Model & Visualization (Google Colab / Python)
This section demonstrates the machine learning workflow using Python within a Google Colab notebook, connecting directly to the feature-engineered data in Snowflake.

(Click the "Open In Colab" badge above to view and run the notebook interactively.)


**2.1. Data Loading & Preprocessing**
The snowflake-connector-python library is used to securely connect to Snowflake (using Colab's userdata secrets manager) and pull the FEATURES.CUSTOMER_CHURN_FEATURES table into a Pandas DataFrame.


**Feature Scaling**: Numerical features are scaled using StandardScaler from scikit-learn. This is a vital preprocessing step for many machine learning algorithms (like Logistic Regression) to ensure features contribute equally regardless of their original scale.

The data is split into training and testing sets (80/20 split) for model development and evaluation.


**2.2. Model Training & Evaluation**
Two different classification models are trained and evaluated:


**Logistic Regression**: A simple, interpretable linear model used as a baseline.


**Random Forest Classifier**: A more complex, non-linear ensemble model, often robust to various data types and capable of capturing intricate relationships.


**Evaluation Metrics:**


**Classification Report:** Provides Precision, Recall, F1-score for each class.


**Confusion Matrix:** Shows the counts of True Positives, True Negatives, False Positives, and False Negatives.


**ROC AUC Score:** Measures the model's ability to distinguish between churned and non-churned customers.


**2.3. Hyperparameter Tuning**
GridSearchCV is used to perform basic hyperparameter tuning for the Random Forest model. This process systematically searches for the best combination of model parameters (e.g., n_estimators, max_depth) to optimize performance (measured by ROC AUC in this case). This demonstrates an understanding of improving model effectiveness.


**2.4. Visualizations**
Key visualizations are generated using matplotlib and seaborn to interpret the data and model performance:


# Receiver Operating Characteristic (ROC) Curve

**Interpretation**: This plot illustrates the diagnostic ability of both models. An Area Under the Curve (AUC) of 1.00 for both Logistic Regression and Random Forest indicates perfect classification in this small, simulated dataset. In real-world scenarios, an AUC closer to 1.00 signifies a strong predictive model.

<img width="891" height="691" alt="image" src="https://github.com/user-attachments/assets/4880f250-53e1-40e4-bdbd-87915d228b2c" />



# Top 10 Feature Importance (Random Forest)
**Interpretation**: This chart highlights the most influential features for churn prediction according to the Random Forest model. TENURE_MONTHS (customer tenure) is identified as the most important feature, followed by PAYMENT_METHOD_ELECTRONIC_CHECK and TOTAL_CHARGES. This provides actionable insights into key drivers of churn.


<img width="1138" height="670" alt="image" src="https://github.com/user-attachments/assets/6ce7d3a3-4bad-49d0-bbbf-ca5462141fc2" />

# Distribution of Churn (True vs. False)


**Interpretation:** This bar chart displays the class balance of the target variable (CHURN_LABEL), showing 6 customers who did not churn and 4 who did. 
Understanding class distribution is crucial for selecting appropriate evaluation metrics and potential handling of class imbalance in larger datasets.


<img width="917" height="626" alt="image" src="https://github.com/user-attachments/assets/aaeb2f01-afb2-43cf-9673-b7ebf3738d39" />


# 3. Conclusion
This project successfully establishes a robust and scalable data pipeline on Snowflake, demonstrating end-to-end data warehousing principles from raw ingestion to feature engineering and analytical reporting. By integrating with Google Colab and Python, the project further showcases a comprehensive data science workflow, including essential steps like feature scaling, training and evaluating multiple machine learning models (Logistic Regression and Random Forest), performing hyperparameter tuning, and generating insightful visualizations.

**For a data science team, this project provides:**
The machine learning component offers the capability to proactively identify at-risk customers, allowing the team to implement targeted retention strategies.

The feature importance analysis provides clear, actionable insights into which customer attributes most influence churn, guiding product development and marketing efforts.



# 4. Recommendations & Future Work
Based on this foundational work, here are key recommendations and areas for future development:

**Integrate with a Larger, Real-world Dataset:** Apply this pipeline to a more extensive and diverse dataset to build a more robust and generalizable model. This would also allow for more nuanced insights from the analytical queries and model evaluation.


**Advanced Feature Engineering:** Explore more complex feature interactions (e.g., cross-features), time-series features (if historical data is available), or external data sources (e.g., customer service interactions, website usage data) to enrich the feature set.


**Explore More Sophisticated Models**: Experiment with deep learning models (e.g., LSTMs for sequential data) or advanced boosting algorithms (e.g., LightGBM, CatBoost) that might capture more complex patterns.


**Model Deployment & Monitoring:** Implement a strategy for deploying the trained model (e.g., as an API endpoint using Flask/FastAPI, or via Snowpark for in-Snowflake inference) and continuously monitoring its performance in production to detect data drift or concept drift. This is crucial for maintaining model effectiveness over time.


**A/B Testing Integration:** If applicable, design and analyze A/B tests for churn prevention strategies, using the model's predictions to target specific customer segments for interventions.


**Automated Retraining Pipeline**: Develop an automated pipeline for periodic model retraining to ensure the model remains up-to-date with changing customer behaviors and market conditions.
