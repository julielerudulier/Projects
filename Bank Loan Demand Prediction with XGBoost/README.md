## Bank Loan Demand Prediction

In this project, I used meta learner XGBoost to enhance the performance of two basic learners (Linear Regression and Decision Tree) to predict which customers would be most like to get a bank loan. 

#### Language:
- Python

#### Librairies used in this project:
- Numpy
- Pandas
- Matplotlib Pyplot
- Seaborn
- Scikit-learn
- XGBoost
  
#### Data:
For this project I used a dataset which contains data on 5,000 customers of a bank. The data include customer demographic information (age, income, etc.), the customer's relationship with the bank (mortgage, securities account, etc.), and the customer response to the last personal loan campaign (Personal Loan). The dataset is available for download [here](https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling).

#### Outcome:
I was able to get very high scores with the base learners only, the Decision Tree model especially, with an accuracy score of 0.9880. But the meta learner still had a positive effect on our model, with an overall precision of 0.9890 for the customers without a loan, and 0.9773 for those with a loan. 
