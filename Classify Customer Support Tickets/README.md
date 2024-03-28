## Classify Customer Support Tickets

In this project I built a model to classify support tickets using Natural Language Processing and a Multinomial Naive Bayes model.

#### Language:
- Python

#### Librairies used in this project:
- Pandas
- Numpy
- sklearn

#### Data:
For this project I used Microsoft's anonymised support ticket dataset, available for download [here](https://github.com/vat3300/SSMA-Smart-Service-Management-Assistent-/blob/main/all_tickets.csv).

#### Outcome:
I was able to show that customers on the international plan are significantly more likely to churn. Obviously, there's also a link with the total international charges metric, so perhaps they're leaving because the prices are too high.

The next big issue is with the number of 'customer of service' calls placed. It seems important to identify what went wrong and caused these customers to call.

Finally, customers on the voice mail plan are least likely to churn, so there might be something the company can do here. Perhaps aiming to acquire voice mail customers, encouraging others to move to this plan, or examining why the customers here are less likely to churn than the rest would help.
