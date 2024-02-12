# syriatel_churn

SYRIATEL CUSTOMER CHURN PREDICTION

PROJECT OVERVIEW: 

In Syria, the telecommunications industry faces a significant challenge in retaining customers amidst increasing competition and evolving consumer preferences. SyriaTelcom, one of the leading telecom service providers in the country, seeks to reduce customer churn by identifying patterns and factors contributing to customer attrition. High customer churn not only results in revenue loss but also undermines the company's reputation and market position.

![What-are-5G-Cell-Towers](https://github.com/Saoke1219/syriatel_churn/assets/144167777/4063cd6c-72d9-4b5a-a2f8-a0a6d1775eeb)


### BUSINESS PROBLEM OBJECTIVE:

SyriaTel, a telecommunications company, aims to proactively address customer churn to retain valuable customers, reduce revenue loss, and enhance overall customer satisfaction and loyalty. To achieve this objective, SyriaTel seeks to develop a predictive model capable of identifying customers at risk of churn. By leveraging historical customer data and predictive analytics, SyriaTel aims to anticipate potential churn events and implement targeted retention strategies to mitigate churn and foster long-term customer relationships.


### DATA UNDERSTANDING:

The dataset used in this project was obtained from SyriaTelcom's internal database, which contains comprehensive records of customer interactions and telecommunications services(+3000 customers and 20 columns).This makes it highly suitable for addressing the business problem at hand of predicting customer churn for Syria Telcom.


### METHODS:
The methods used are;
1. Logisctics model
2. Decision Tree model
3. Random Forest model

For Decision Trees,Random Forests Models and Logistic Regression; we alculated accuracy using the confusion matrix
Evaluated Precison, Recall, and f1 score of each model.
 We also carried out hyperparemeter tunning of the best performing model(Random forest). 
The outcome was a precision_score was 96%.

## ROC Curve

![ROC_Curve](https://github.com/Saoke1219/syriatel_churn/assets/144167777/903a650f-23ce-46ae-97ed-7d0e76cf7f48)


The ROC curve is a plot of the true positive rate against the false positive rate of our classifier. The best performing models will have a curve that hugs the upper left of the graph, which is the the random forest classifier in this case.

RECOMMENDATIONS:

1.Focus on Total Day Charge and Total Evening Charge:

Features such as "total day charge" and "total eve charge" have significant importance in predicting churn. The company should analyze pricing strategies for daytime and evening usage, ensuring they are competitive and aligned with customer expectations. Consider offering customizable plans or incentives to reduce charges during peak hours.

2.Improve Customer Service Quality:

"Customer service calls" emerged as a crucial predictor of churn. Enhance customer service quality by investing in training, technology, and support resources. Proactively address customer issues and complaints to minimize the need for repeated service calls, ultimately improving customer satisfaction and retention.

3.Encourage Voice Mail Plan Adoption:

While "voice_mail_plan_is_yes" has moderate importance, it still contributes to predicting churn. Develop targeted marketing campaigns to promote voice mail plan adoption among customers. Highlight the benefits of voice mail services, such as message storage and accessibility, to increase their perceived value and encourage uptake.

4.Optimize International Calling Services:

Features related to international calling, such as "total intl calls" and "total intl charge," exhibit some importance. Review international calling rates, explore partnerships with global carriers, and introduce cost-effective international calling plans or bundles to attract and retain customers who frequently make international calls.

5.Address Area Code Specific Concerns:

While area code features have relatively low importance, they still contribute to predicting churn. Conduct targeted surveys or customer outreach to identify any area-specific issues or preferences. Tailor marketing strategies or service offerings to address the unique needs of customers in specific geographic areas, potentially improving customer satisfaction and loyalty.

## CONCLUSION:

In conclusion, the project provides a solid foundation for the telecommunications company to develop and implement data-driven strategies aimed at reducing churn, improving customer satisfaction, and ultimately driving business growth.

### NEXT STEPS:
We can improve the model accuracy by establishing a process for continuous monitoring and maintenance of the model once deployed. Regularly evaluate its performance, update it with new data as it becomes available, and refine it based on evolving business needs and customer behaviors.












    













