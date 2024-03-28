## Causes of Customer Churn

In this project I used Cox’s Proportional Hazards model to examine a contractual churn dataset to quickly identify the factors associated with the loss of customers. 

#### Language:
- Python

#### Librairies used in this project:
- Pandas
- Numpy
- CoxPHFitter
- Matplotlib.pyplot

#### Data:
For this project I used Kaggle's Customer Churn 2020 dataset, which is based on a telecoms provider and includes details on the location of the customers, the telephone plan they have, how they use it, how much they’re being charged, and how many times they’ve had to call the customer service department. The dataset is available for download [here](https://www.kaggle.com/competitions/customer-churn-prediction-2020/data).

#### Outcome:
I was able to show that customers on the international plan are significantly more likely to churn. Obviously, there's also a link with the total international charges metric, so perhaps they're leaving because the prices are too high.

The next big issue is with the number of 'customer of service' calls placed. It seems important to identify what went wrong and caused these customers to call.

Finally, customers on the voice mail plan are least likely to churn, so there might be something the company can do here. Perhaps aiming to acquire voice mail customers, encouraging others to move to this plan, or examining why the customers here are less likely to churn than the rest would help.
