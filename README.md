# Nonprofit Networking
## Overview
### *Purpose*
Alphabet Soup is a non-profit foundation dedicated to helping organizations that protect the environment, improve people’s well-being, and unify the world, and has raised over $10 billion dollars in the last 20 years. This money has been used to invest in life-saving technologies and organize reforestation groups around the world. Beks, a Data Scientist and Programmer for Alphabet Soup, is responsible for analyzing the impact of each donation and examining potential recipients to ensure the company’s money is being used effectively. The President of Alphabet Soup has asked Beks to determine which organizations are worth donating to and which are too high of risk by creating a mathematical data-driven solution. Beks has a CSV containing over 34,000 organizations that Alphabet Soup has provided funding to and has decided to use this dataset to design and train a deep learning neural network. The following variable(s) were included in the dataset:
•	EIN and NAME—Identification columns
•	APPLICATION_TYPE—Alphabet Soup application type
•	AFFILIATION—Affiliated sector of industry
•	CLASSIFICATION—Government organization classification
•	USE_CASE—Use case for funding
•	ORGANIZATION—Organization type
•	STATUS—Active status
•	INCOME_AMT—Income classification
•	SPECIAL_CONSIDERATIONS—Special consideration for application
•	ASK_AMT—Funding amount requested
•	IS_SUCCESSFUL—Was the money used effectively

## Results
### *Data Preprocessing*
1.	What variables were considered the target for the model?
The “IS_SUCCESSFUL” variable was the target of the model as it identified which companies used the donations effectively.
2.	What variable(s) were considered to be the features for the model?
The “APPLICATION_TYPE”, “AFFILIATION”, “CLASSIFICATION”, “USE_CASE”, “ORGANIZATION”, “STATUS”, “INCOME_AMT”, “SPECIAL_CONSIDERATIONS”, and “ASK_AMT” variables appear to be the features for the model.
3.	What variable(s) were considered neither targets nor features, and should be removed from the input data?
The “EIN” and “NAME” variables were identified to be neither targets nor features and were dropped from the application_df DataFrame. 
### *Compiling, Training, and Evaluating the Model
1.	How many neurons, layers, and activation functions did you select for your neural network model, and why?

I created a neural network model by assigning the number of input features to the length of the X_train_scaled data, set the nodes of the first layer to 80, and set the nodes of the second layer to 30. For the first and second hidden layer, I set the activation parameter to the ReLU activation function to identify and train on nonlinear relationships in the dataset, while the activation parameter for the output layer was set as the sigmoid activation function to produce a probability output. 

2.	Were you able to achieve target model performance?

![model1_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model1_loss_accuracy.png)

The model had an accuracy of 72.6% and a loss of 57.0%, and thus I was unable to achieve the target model performance of 75%. 

3.	What steps did you take to try and increase model performance?

To optimize model performance, I first identified any columns with a high number of unique values. The “APPLICATION_TYPE”, “CLASSIFICATION”, and “ASK_AMT” columns contained more than 10 unique values so I determined the number of data points for each unique value and created a density plot for each of the value counts. According to the density plot for the “APPLICATION_TYPE” value counts, the most common unique values had more than 500 instances within the dataset and thus any application type that appeared fewer than 500 times in the dataset were put into the bucket “Other”. 

![application_type_plot.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/application_type_plot.png)

Similarly, the density plot for the “CLASSIFICATION” value counts showed that the most common unique values had more than 1000 instances within the dataset and thus any classification that appeared fewer than 1000 times in the dataset were put into the bucket “other”.

![classification_plot.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/classification_plot.png)

The “ASK_AMT” column included over 8,747 unique values and so I calculated and plotted the value counts to determine the potential for binning. 

![ask_amt_count.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/ask_amt_count.png)

![ask_amt_plot.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/ask_amt_plot.png)

The “ASK_AMT” value counts ranged from the ask amount of $5000 with the highest value count at 25,398 to the ask amount of $6,948,863 with a value count of 1. Due to vast difference in asking amounts and value counts, I removed the “ASK_AMT” column from the DataFrame.

![model2_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model2_loss_accuracy.png)

Removing the “ASK_AMT” column slightly did not change the accuracy of the model, as it remained at 72.6%. 

I further attempted to optimize the model to achieve a predictive accuracy greater than 75% by adding 20 additional neurons to the second hidden layer. 

![model3_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model3_loss_accuracy.png)

Adding 20 additional neurons to the second layer slightly increased the accuracy to 72.5%. As only a minor improvement was seen, I additionally added a third hidden layer with 20 neurons.

![model4_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model4_loss_accuracy.png)

Adding an additional hidden layer with 20 neurons slightly reduced the accuracy of the model to 72.7%. So, I reverted the model back to the original with two hidden layers and instead changed the activation function to the tanh function for the hidden layers.

![model5_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model5_loss_accuracy.png)

Changing the activation function of the hidden layers to the tanh function decreased the accuracy slightly to 72.5%. So, I changed the activation functions back to the ReLU function and then added 100 additional epochs to the model with the additional neurons and hidden layer.

![model6_loss_accuracy.png]( https://github.com/kcharb7/Neural_Network_Charity_Analysis/blob/main/Images/model6_loss_accuracy.png)

Adding an additional 100 epochs to the training regimen slightly reduced the accuracy of the model to 72.5%.

## Summary
A DataFrame, application_df, was created using a CSV file containing over 34,000 organizations that Alphabet Soup has provided funding to. The “EIN” and “NAME” variables were identified to be neither targets nor features and were dropped from the application_df DataFrame. The “IS_SUCCESSFUL” variable was the target of the model, while the remaining columns were the features. These columns were used to create and train a neural network model. The neural network model was created with two hidden layers consisting of 80 and 30 nodes, respectively. Each hidden layer used the ReLU activation function to identify and train on nonlinear relationships in the dataset, while the output layer used the sigmoid activation function to produce a probability output. The accuracy of the model was 72.6% and steps were taken to improve the model, including dropping the “ASK_AMT” column from the DataFrame, adding additional neurons to the second hidden layer, adding a third hidden layer, changing the activation functions of the hidden layers, and adding addition epochs. None of these adjustments improved the accuracy of the model. So, it is recommended that a different model be tested. As our target is to classify an organization as successful or non-successful, a binary classifier such as Support Vector Machines (SVMs) is recommended. SVMs can build adequate models with linear or nonlinear data. 
