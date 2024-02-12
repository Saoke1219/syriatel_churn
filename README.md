# syriatel_churn

SYRIATEL CUSTOMER CHURN PREDICTION

PROJECT OVERVIEW: 

In Syria, the telecommunications industry faces a significant challenge in retaining customers amidst increasing competition and evolving consumer preferences. SyriaTelcom, one of the leading telecom service providers in the country, seeks to reduce customer churn by identifying patterns and factors contributing to customer attrition. High customer churn not only results in revenue loss but also undermines the company's reputation and market position.

![What-are-5G-Cell-Towers](https://github.com/Saoke1219/syriatel_churn/assets/144167777/4063cd6c-72d9-4b5a-a2f8-a0a6d1775eeb)

BUSINESS PROBLEM OBJECTIVE:

SyriaTel, a telecommunications company, aims to proactively address customer churn to retain valuable customers, reduce revenue loss, and enhance overall customer satisfaction and loyalty. To achieve this objective, SyriaTel seeks to develop a predictive model capable of identifying customers at risk of churn. By leveraging historical customer data and predictive analytics, SyriaTel aims to anticipate potential churn events and implement targeted retention strategies to mitigate churn and foster long-term customer relationships.

OBJECTIVE:

The objective of this project is to analyze SyriaTelcom's customer data to understand the factors influencing churn and develop predictive models to forecast customer attrition. By leveraging machine learning algorithms and predictive analytics, the project aims to:

Identify key features and patterns associated with customer churn and non-churn.

Build predictive models to forecast the likelihood of churn for individual subscribers.

Provide actionable insights to SyriaTelcom for implementing targeted retention strategies and reducing customer attrition.

Enhance customer satisfaction and loyalty by addressing the underlying issues driving churn.

Improve SyriaTelcom's market position and competitiveness in the telecommunications industry by fostering long-term customer relationships.

RESEARCH QUESTIONS:

1 .What are the key factors contributing to customer churn ?

2 .How do characteristics, such as location, influence the likelihood of customer churn?

3 .Are there specific contract terms or pricing plans associated with higher churn rates among customers?

4 .Which is the best model to accurately predict churn?

Data 

NUMERICAL FEATURES: (account length, number vmail messages, total day minutes, total day calls, total day charge, total eve minutes, total eve calls,total eve charge,total night minutes,total night calls,total night charge,total intl minutes,total intl charge,customer service calls)

CATEGORICAL FEATURES: (state,area code,international plan,voicemail plan)

We will require the following libraries;

```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.impute import SimpleImputer

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV

from sklearn.tree import DecisionTreeClassifier 

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

Seaborn style:

sns.set(style="whitegrid")
```

DATA EXPLORATION:

In this analysis, we will be dropping the 'phone number' column as it is a unique identifier for each customer therefore not relevant for analysis. The 'churn' feature serves as the dependent variable.The 'churn' variable signifies whether a customer has terminated their contract with SyriaTel. A value of 'True' means a contract termination, while 'False' indicates that the customer has not terminated their contract and maintains an active account.

![Churn_Distribution](https://github.com/Saoke1219/churn_analysis/assets/144167777/9fe78271-f2d2-4579-a28f-c0a00727a95c)

The above pie chart shows the distribution of churned and non-churned syria tel customers.The distribution is indicated in percentage,with 14.5% "true churn" indicates customers who have ended their subscription. 85.5% "false churn" indicates customers who are still active subscribers.This also shows "non-churn") has a much higher count compared to the other class ("churn"), indicating that the dataset has a class imbalance.

![numerical_distribution_plot](https://github.com/Saoke1219/churn_analysis/assets/144167777/ea29c2c6-b04d-41e3-90e6-e7294875f705)

Above are distribution plots of churned and non-churned customers in the numerical category.We observe that non-churned customers are more than churned customers.we also observe that the distribution is normal while that of total international calls is skewed to the right though still normally distributed.

![numeric_correlation_heatmap](https://github.com/Saoke1219/churn_analysis/assets/144167777/a7b34a79-6976-4b31-b0d7-2f7b477f4b43)

From the correlation heatmap,some of the features in the dataset demonstrate a perfect positive correlation, such as "Total day charge" and "Total day minutes", "Total eve charge" and "Total eve minutes", "Total night charge" and "Total night minutes", and "Total int charge" and "Total int minutes".They have a correlation coefficient of 1.00, indicating perfect multicollinearity.

explore categorical features

categoric_cols = ['state','area code','international plan','voice mail plan']

### DISTRIBUTION OF CATEGORICAL VARIABLES

![voice_mail_plan_count_on_churn](https://github.com/Saoke1219/churn_analysis/assets/144167777/4141d0b3-1499-409a-a7dc-d8fc8ab594a9)

We can observe from the plot above that there is a significantly low churn rate among customers with a voicemail plan.
This indicates customers have a preference of using this plan.


