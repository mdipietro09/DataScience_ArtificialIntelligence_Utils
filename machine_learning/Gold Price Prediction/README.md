# Gold-Price-Prediction-using-Python
This is a Gold Price Predictor where it predicts the price of gold on the basis of different data like Date, Silver etc. 

Dataset Link : https://www.kaggle.com/datasets/altruistdelhite04/gold-price-data

<h1 align='center'> WORKFLOW OF THE PROJECT<br></h1>

```mermaid
flowchart TD

A[Step 0 : Collect Data] --> B[Step 1 : Import Libraries/Modules in the workspace]
B[Step 1 : Import Libraries/Modules in the workspace] --> C[Step 2 : Import the collected data into the workspace]
C[Step 2 : Import the collected data into the workspace] --> D[Step 3 : Data Preprocessing]
D[Step 3 : Data Preprocessing] --> E[Step 4 : Perform EDA by visualizing the data]
E[Step 4 : Perform EDA by visualizing the data] --> F[Step 5 : Train ML model using RANDOM FOREST REGRESSOR]
```
  
  
<ol>
  <li><b><i>DATA COLLECTION</i></b> - The SONAR data used in the above project is collected from kaggle. <br>Link : https://tinyurl.com/ybm7fpwp<br>
  <li><b><i>SETTING UP WORKSPACE/ENVIRONMENT</i></b> - This basically means importing of all the required modules and importing the data.<br>
  Modules used in this project : 
  
  ```
    import numpy as np                                          #to convert data into numpy arrays
    import pandas as pd                                         #for data pre-processing technique and importing our data
    import matplotlib.pyplot as plt                             #for creating data visualizations to explore the data
    import seaborn as sns                                       #for making such visualizations and creating plots
    from sklearn.model_selection import train_test_split        #to divide our original data into training data and testing data
    from sklearn.ensemble import RandomForestRegressor          #to build our regression type ml model from sklearn import metrics #to evaluate the accuracy of our ml model
  ```
  
  <li><b><i>DATA PRE-PROCESSING</b></i> - This step involves the set of different processes like data cleaning, data integration and other such processes which basically means removing noise and inconsistency from the data.
    
  <li><b><i>TRAIN TEST SPLIT</b></i> - This is the crucial step where we divide the dataset into two halves called "Training data" and "Testing data". This helps to test the accuracy score of the model which going to be developed and trained.
    
   ```
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=2)
   ```
  
 <li><b><i>RANDOM FOREST REGRESSOR</b></i> - This is a main model which we were talikng about. We chose "Random Forest Regression Model out of all other models because we need a model which predict a proper answer in numerals rather than a classified type of result."
</ol>

    
###  In this way we can predict the price of gold. The prediction can be done more accurately if used neural networks and huge dataset and deep learning .
