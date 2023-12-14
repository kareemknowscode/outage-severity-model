#

Authors: Mariana Paco, Kareem Mazboudi

This is a data science project that aims to utilize machine learning models for predicting the severity of power outages. Our previous exploratory data analysis on this dataset can be found [here](https://kareemknowscode.github.io/ca-economy-outages/). Our findings are presented in this report, which was completed as part of the DSC80 course at UCSD

## Introduction and Framing the Problem
Anticipating and mitigating power outages is key to ensuring the stability of public welfare and infrastructure. We aim to use data regarding power outages around the United States to predict the severity of future outage events and enable proactive measures for citizens who may be impacted. 

In this model, we use a Random Forest regression model to predict our response variable: the proportion of citizens affected by power outage events. We selected this variable for a few reasons, with the main one being ease of interpretation. We believe that if we can predict the proportion of those affected by outages, we can directly apply this to predicting outage severity given other factors such as the cause of any given outage, climate, region, and the ONI (Oceanic Niño Index). This model uses RMSE and R<sup>2</sup> as the main evaluation metrics to determine accuracy.

To make our model robust, we only use the features listed above given that the data could only be obtained from previous power outage events and are not dependent on outside factors such as actual weather events at the time. 
