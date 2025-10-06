## Infer the effects of marketing

In this project I used Google's CausalImpact to try identify whether a marketing campaign/action had a statistically significant impact on a chosen metric.

#### Language:
- Python

#### Librairies used in this project:
- Pandas
- Causalimpact

#### Data:
For this project I used a time series dataset based on Google Search Console data where a change was made after July 17th 2021. The dataset is available for download [here](https://raw.githubusercontent.com/flyandlure/datasets/master/causal_impact_dataset.csv).

#### Outcome:
For the given dataset, the model predicted that in the absence of any intervention we should have expected to see an average of 243 clicks in the post-intervention period, but we actually generated 344.
It thus seems reasonable to think that this corresponds to the causal effect the intervention had upon the response variable - the clicks. The site changes made increased clicks by nearly 42%, which was statistically significant and is unlikely to be random (but may, of course, have been caused by something else).
