## Churn Model with XGBoost

In this project I built a customer retention model for a contractual setting and aimed to predict which customers were retained and which would be most likely to churn based on the experience they’ve had as customers of a phone provider. The model examines how much they’ve used the service, when they use it, how much they’ve been charged, and whether they’ve been in touch with customer service. I also used XGBoost and Optuna for hyperparameter tuning.

#### Language:
- Python

#### Librairies used in this project:
- Pandas
- Sklearn
- XGBoost
- Optuna

#### Data:
For this project I used Kaggle's Customer Churn 2020 dataset, which is based on a telecoms provider and includes details on the location of the customers, the telephone plan they have, how they use it, how much they’re being charged, and how many times they’ve had to call the customer service department. The dataset is available for download [here](https://www.kaggle.com/competitions/customer-churn-prediction-2020/data).

#### Outcome:
I was able to get back an accuracy score of 97.6% on the test dataset, with an AUC score of 0.92. The tuned model performed better than the untuned model and I was eventually able to predict with very high accuracy which of customers would churn and which would be retained, purely based on their tenure, call plan, call charges and usage, and how often they needed to contact customer service.
