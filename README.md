# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
 # FEATURE SCALING
 import pandas as pd
 from scipy import stats
 import numpy as np
```
```
df=pd.read_csv("bmi.csv")
df.head()
```
<img width="406" height="243" alt="image" src="https://github.com/user-attachments/assets/a1f774f7-255c-47de-b555-eb2500ddda19" />

```
 df_null_sum=df.isnull().sum()
 df_null_sum
```
<img width="450" height="119" alt="Screenshot 2025-10-08 114751" src="https://github.com/user-attachments/assets/6ef73573-217b-4eda-86b0-979df226b421" />

```
df.dropna()
```
<img width="435" height="498" alt="image" src="https://github.com/user-attachments/assets/e69a8b16-84dc-4747-8446-9ee85029353a" />

```
 max_vals = np.max(np.abs(df[['Height', 'Weight']]), axis=0)
 max_vals
```
<img width="466" height="72" alt="image" src="https://github.com/user-attachments/assets/8fe67515-f6e0-4b79-b0bc-1d2f17cfb04e" />

```
 from sklearn.preprocessing import StandardScaler
 df1=pd.read_csv("bmi.csv")
 df1.head()
```

       
# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file has been executed successfully. 
