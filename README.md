# Predicting the Severity of Power Outages: Predictive Modeling and Analysis

Authors: Mariana Paco, Kareem Mazboudi

This is a data science project that aims to utilize machine learning models for predicting the severity of power outages. Our previous exploratory data analysis on this dataset can be found [here](https://kareemknowscode.github.io/ca-economy-outages/). Our findings are presented in this report, which was completed as part of the DSC 80 course at UCSD

## Introduction and Framing the Problem
Anticipating and mitigating power outages is key to ensuring the stability of public welfare and infrastructure. We aim to use data regarding power outages around the United States to predict the severity of future outage events and enable proactive measures for citizens who may be impacted. 

In this model, we use a Random Forest regression model to predict our response variable: the proportion of citizens affected by power outage events. We selected this variable for a few reasons, with the main one being ease of interpretation. We believe that if we can predict the proportion of those affected by outages, we can directly apply this to predicting outage severity given other factors such as the cause of any given outage, climate, region, and the ONI (Oceanic Niño Index). This model uses RMSE and R<sup>2</sup> as the main evaluation metrics to determine accuracy.

To make our model robust, we only use the features listed above given that the data could only be obtained from previous power outage events and are not dependent on outside factors such as actual weather events at the time. 

## Baseline Model
As stated previously, we are using a random forest regression model to make our predictions from the data. At the time from the start of our prediction, we will use a variety of features from the dataset that we believe to be useful in predicting outage severity, listed below:

### Features

| **Variables**             | **Description**                                              |
|---------------------------|--------------------------------------------------------------|
| `'YEAR'`                  | Year that the outage occurred                                |
| `'MONTH'`                 | Month that the outage occurred                               |
| `'U.S._STATE'`            | State that the outage occurred                                |
| `'NERC.REGION'`           | Categorical data describing the climate at the time of the outage|
| `'CLIMATE.REGION'`        | NCE Climate Regions for a given U.S. state                   |
| `'CLIMATE.CATEGORY'`      | Climate episode for a given U.S. state                        |
| `'CAUSE.CATEGORY'`        | Categorical data describing the cause of the outage, e.g., “severe weather” |
| `'PC.REALGSP.STATE'`      | Per capita real GSP of a given state (adj. for inflation, 2009 chained $USD) |
| `'TOTAL.CUSTOMERS'`       | The annual number of total customers served in a U.S. state   |
| `'ANOMALY.LEVEL'`         | Oceanic Niño Index. Scores of +0.5 and higher indicate El Niño. Scores of -0.5 and lower indicate La Niña.|

In total, we have 15 features for our baseline model which will be useful in generating accurate predictions regarding power outage severity. `'U.S._STATE'`, `'NERC.REGION'`, `'CLIMATE.REGION'`, `'CLIMATE.CATEGORY'`,  `'CAUSE.CATEGORY'`, and `'CAUSE.CATEGORY.DETAIL'` are our nominal features, which we will be one hot encoding for our model. `'YEAR'` and `'MONTH'` are technically ordinal features, but to predict severity, we will treat them as nominal rather than ordinal since there is no reasonable trend to follow in terms of going year by year. 

Our quantitative features include `'PC.REALGSP.STATE'`, `'CUSTOMERS.PROPORTION'`, `'TOTAL.CUSTOMERS'`, `'ANOMALY.LEVEL'`, `'OUTAGE.DURATION'`. We will be standardizing these features to gauge variation among the data and judge just how severe any given outage event is relative to the average severity of outages from our data.

`'CUSTOMERS.PROPORTION'` is not a feature included in the original dataset. We created it by dividing the customers affected by an outage by the total customer count of the US state. This is going to be our response variable, which will greatly help in quantifying the severity of a power outage in terms of how widespread it might be. We expect that larger outages will increase the proportion of customers that experience outages, which we believe justifies our use of this proportion as our response variable. 

Our data cleaning process was nearly the same as we had performed previously in our analysis of the effects of outages on California’s GSP, except that now we will include all 50 states and the NERC regions that they are associated with. We also include US climate regions as well. In terms of missing data, we mostly dealt with these issues by dropping the rows that did not contain data given that it was not possible to impute that data, such as a missing outage restoration date or a missing cause category. For the rows with missing data we did not up and drop, we were able to determine the correct values that we then imputed. For example, the state of Hawaii did not originally have any values in the `'CLIMATE.REGION'` column, and we imputed its region as the Pacific Islands region. We were able to impute some values in `'CAUSE.CATEGORY.DETAIL'` by duplicating self-explanatory values from the `'CAUSE.CATEGORY'` column such as “islanding”, which describes the exact issue that caused the outage to occur. 
Using this data, we separated our model into training and testing sets and ran the model. We used an 80-20 split and a max depth of 5 for our random forest regressor and from this, we got the following metrics on average (rounded to the nearest hundredth):

### Results

|**Training Data**				|**Test Data**|
|R<sup>2</sup> = 0.71     |R<sup>2</sup> = 0.21|
|RMSE = 0.05              |RMSE = 0.04|

Our model achieved a 21% accuracy on average for predictions made from the given test set. This is somewhat weak for our initial model but given that we have not yet optimized our hyperparameters or added more advanced features to our model, we are somewhat satisfied. More importantly, we can see from how close the training and testing RMSEs are that our model generalizes very well and thus shows that our model may not be prone to overfitting.

## Final Model 
In our final model, we aim to improve upon our baseline by optimizing hyperparameters and adding new features that we believe will make our model more robust. The new features that we added are as follows:

|**Features**				|**Description**|
|`'RES.PRICE'`      |Standardized monthly electricity price for the residential sector in cents/kW-H|
|`'elnino'`/`'lanina'`              |Binarized columns containing 1 if true for El Niño/La Niña anomaly event(t=±0.05)|
|`'TOTAL.SALES'` |Quartiles of total electricity consumption in a given U.S. state|

Previous categorical and numerical features we used in the baseline model were maintained. A new squared `'ANOMALY.LEVEL'` was created to try and capture the variation of Niño anomaly events, and `'PC.REALGSP.STATE'` was turned into quartiles to gauge variation around the median level of real GSP per state. The main reason we selected these features is because we believe that electric company data about how many customers they serve in the residential sector is important in predicting the area-wide severity of outages. The intuition behind specifically focusing on the proportions of residential customers is simple: If most land is occupied by residences, then we can likely accurately predict the proportion of customers affected by an outage event using this data. 

We experimented with various supervised regression algorithms for our final model. One key one we wish to mention, but did not use was a LASSO regression model. We wanted to find the features that had the strongest relationship with our response variable. The large number of features that we had may have been causing noise in our predictions, and we felt that LASSO was most advantageous in dealing with features that might not have any effect on our response variable. We ended up deciding on Random Forest regression since we wanted to minimize features and use engineered features that would help capture hidden relationships in the data, such as squared anomaly level. We also determined an overfit in the RR<sup>2</sup> of the LASSO model, which was not sustainable in practice.

Our final model again uses a Random Forest regression model to make predictions. Along with this, we have implemented Grid Search as our cross-validation algorithm to optimize our hyperparameters, specifically max tree depth = 5, the minimum number of samples before splitting a leaf = 200, and the criterion for which the model evaluates data was selected to be in terms of the Poisson distribution. Given the feature selection and the hyperparameter optimization, these are our model’s evaluation metrics:

|**Training Data**				|**Test Data**|
|R<sup>2</sup> = 0.45     |R<sup>2</sup> = 0.25|
|RMSE = 0.05              |RMSE = 0.04|

The test set accuracy has increased by 4% on average, up to 25%, showing improvement in terms of our model’s prediction power. Not only that but the RMSEs of the training and test set are still relatively close, and this maintains our model’s high generalization power which was achieved in the baseline model. Overall, this is an improvement over our baseline model. We had hoped to predict with much more accuracy, but we believe that the low accuracy can be attributed to a few key facts. For one, this model predicts power outage severity across the ENTIRE country by NERC region and state. Given that each region has its unique power grid, likely with different technical specifications, it is difficult to ascertain the various outside factors that might influence the model’s performance in this case. 

## Fairness Analysis

Our assessment of fairness will rely on two groups from the dataset. We will be checking whether or not samples of RMSEs from the WECC region (Western Electric Coordinating Council) and the TRE (Texas Reliability Entity) are drawn from the same distribution. In other words, we are checking the West Coast's power grid against Texas's power grid, and using their predicted RMSEs. Specifically, we are using the absolute difference in RMSEs for our test statistic and we will be conducting this test at the 5% level.

**Null Hypothesis/H0**: Our model’s RMSEs for the WECC region and the TRE region are nearly the same, implying fairness in our prediction model.

**Alternative Hypothesis/H1**: Our model’s RMSEs for the WECC region and the TRE region are not the same, implying that our prediction model unfairly predicts across groups.

After conducting the test, we determined a p-value of `p = 0.93`, which is not statistically significant at the 5 percent level. We fail to reject our null hypothesis. 

We conclude that our model is likely to be fair and that any differences in RMSEs are due to random chance. In terms of generalizability, our model will likely always be usable on any dataset with the same data-generating process.
