## Credit Fraud Detection

In this project, I built various predictive models to see how accurate they are in detecting whether a transaction is a normal payment or a fraud.

#### Language:
- Python

#### Librairies used in this project:
- Pandas
- Numpy
- Tensorflow
- Keras (Activation, Dense, Categorical_crossentropy, Sequential, Adam)
- Imbalanced-learn (Classification_report_imbalanced, SMOTE, Make_pipeline, NearMiss)
- Scikit-learn (PCA, TruncatedSVD, RandomForestClassifier, LogisticRegression, TSNE, accuracy_score, DecisionTreeClassifier, SVC, KNeighborsClassifier, StandardScaler...)
- SciPy Stats (norm)
- Matplotlib
- Seaborn
- Time
- Collections
- Itertools

#### Data:
For this project I used a fraud detection dataset which contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where 492 transactions out of 284,807 total transactions were actually frauds. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. The dataset is available for download [here](https://www.kaggle.com/datasets/whenamancodes/fraud-detection?resource=download).

#### Outcome:
I was able to build an under-sampling model with an accuracy score of 0.952 and an over-sampling model using the SMOTE technique with an accuracy score of 0.986.
