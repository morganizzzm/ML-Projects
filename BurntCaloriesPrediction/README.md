Burnt Calories Prediction Project
---------------------------------

A machine learning project that predicts the number of calories burned based on user data and activities.

-----------------------------------------------------------
The Burnt Calories Prediction Project is a machine learning application that estimates the number of calories a person 
burns based on their personal data and physical activities. It uses a predictive model trained on a dataset of individuals 
with their activity levels and corresponding calorie expenditure.

------------------------------------------------------------
Features
------------------------------------------------------------

Predicts burnt calories based on user inputs.
Provides insights into factors affecting calorie burn.
Easy-to-use user interface (if applicable).

------------------------------------------------------------    

Getting Started
-----------------------------------------------------------

Follow these instructions to set up and run the Burnt Calories Prediction Project on your local machine.
                           
Installation
Clone the repository to your local machine:
git clone https://github.com/morganizzzm/burnt-calories-prediction.git

Change into the project directory:
cd burnt-calories-prediction

To use the Burnt Calories Prediction Project:
change the values in the last cell to values you want to make a prediction for
pay attention that the data should be of the form <'Age','Duration', 'Heart_Rate','Body_Temp', 'BMI'>
                            
Run the cell:
sample = np.array([[18,78, 94,40, 21]])
model.predict(sample)

if you don't know your BMI you can use function def calculate_my_BMI(weight, height) function.

-----------------------------------------------------------------

                            
Data Sources
----------------------------------------------------------------
The data is taken from Kaggle
https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos/data

-------------------------------------------------------------------
Data Preprocessing
--------------------

Merge the Data:
Since these datasets are connected by a common field, user_id, you may want to merge them into a single DataFrame for analysis.

Feature Engineering:
We added the new feature: BMI (Body Mass Index) using the weight and height columns by the formula -- 
weight / ((height / 100) ** 2)

Data Splitting
We split our preprocessed dataset into two subsets: the training set and the testing set. 
The purpose of this split is to train the model on one part of the data and evaluate its performance on another part. 
In this code snippet, we use the train_test_split function from the scikit-learn library to perform the split.

Model Training with XGBoost
We choose the XGBoost algorithm, which is a powerful gradient boosting machine learning technique that 
is well-suited for regression tasks. We specify hyperparameters such as the number of estimators (trees), maximum depth of the trees, 
learning rate (eta), subsample ratio, and column subsampling ratio.

Model Prediction
After training the model, we use it to make predictions on the testing set.


--------------------------------------------------------------------                            
Evaluation
-------------------------------------------------------------------

Finally, we evaluate the model's performance using a commonly used metric for regression tasks: the Mean Absolute Error (MAE)
The results for the model on the test is:
Mean Squared Error: 1.9890491606195768


![Unknown-2](https://github.com/morganizzzm/ML-Projects/assets/89296464/089048ab-d413-49eb-9517-224f00be74c3)

