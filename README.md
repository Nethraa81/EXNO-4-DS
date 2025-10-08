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

<img width="389" height="230" alt="image" src="https://github.com/user-attachments/assets/9cf00f8c-4381-42ad-9303-ff019d1c068a" />

```
sc=StandardScaler()
df1[['Height','Weight']]=sc.fit_transform(df1[['Height','Weight']])
df1.head(10)
```
<img width="404" height="413" alt="image" src="https://github.com/user-attachments/assets/8c9d14a3-3af8-4c81-b488-707a8dad8bc6" />

```
 #MIN-MAX SCALING:
 from sklearn.preprocessing import MinMaxScaler
 scaler=MinMaxScaler()
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df.head(10)
```
<img width="396" height="414" alt="image" src="https://github.com/user-attachments/assets/646b745f-c675-4e87-bb30-c9deae152f70" />

```
 #MAXIMUM ABSOLUTE SCALING:
 from sklearn.preprocessing import MaxAbsScaler
 scaler = MaxAbsScaler()
 df3=pd.read_csv("bmi.csv")
 df3.head()
```
<img width="373" height="230" alt="image" src="https://github.com/user-attachments/assets/7ff8e0a3-135a-49e3-9526-8a6af0f55578" />

```
 df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
 df
```
<img width="456" height="491" alt="Screenshot 2025-10-08 133939" src="https://github.com/user-attachments/assets/6fd82613-9d10-4dde-b66e-52b8f99f7024" />

```
 #ROBUST SCALING
 from sklearn.preprocessing import RobustScaler
 scaler = RobustScaler()
 df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
 df3.head()
```
<img width="400" height="220" alt="image" src="https://github.com/user-attachments/assets/9d28350e-faf1-4328-a4ac-089ffb002084" />

```
 #FEATURE SELECTION:
 df=pd.read_csv("income(1) (1).csv")
 df.info()
```
<img width="428" height="407" alt="image" src="https://github.com/user-attachments/assets/81bf1b44-776e-4f87-b8bb-4aa482caba01" />

```
df_null_sum=df.isnull().sum()
df_null_sum
```
<img width="293" height="302" alt="image" src="https://github.com/user-attachments/assets/a20b81d7-a8d1-4bb2-91c0-878d1832f2e9" />

```
 # Chi_Square
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
<img width="986" height="499" alt="image" src="https://github.com/user-attachments/assets/b187615b-b4c0-4cfb-86d3-bbff88476d5a" />

```
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="869" height="504" alt="image" src="https://github.com/user-attachments/assets/0e47f479-5c8b-44d2-bc46-3c6a66f40cd8" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
```
```
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
 from sklearn.ensemble import RandomForestClassifier
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
<img width="390" height="105" alt="image" src="https://github.com/user-attachments/assets/1a6de019-8efd-44a2-aaa6-2c7513e504d3" />

```
 y_pred = rf.predict(X_test)
```
```
 df=pd.read_csv("income(1) (1).csv")
 df.info()
```
<img width="448" height="417" alt="image" src="https://github.com/user-attachments/assets/9e73258c-6eca-49ad-b308-537d135ea5ea" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 categorical_columns = ['JobType', 'EdType', 'maritalstatus', 'occupation', 'relationship', 'race', 'gender', 'nativecountry']
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns]
```
<img width="1027" height="481" alt="image" src="https://github.com/user-attachments/assets/7496ecd4-d7f7-4232-aaed-249144f4b8db" />

```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```
<img width="920" height="501" alt="image" src="https://github.com/user-attachments/assets/84897c36-ee2c-43bc-a0ed-0966f84cc322" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_chi2 = 6
 selector_chi2 = SelectKBest(score_func=chi2, k=k_chi2)
 X_chi2 = selector_chi2.fit_transform(X, y)
 selected_features_chi2 = X.columns[selector_chi2.get_support()]
 print("Selected features using chi-square test:")
 print(selected_features_chi2)
```
<img width="703" height="88" alt="image" src="https://github.com/user-attachments/assets/23b3ddcc-e2e2-4741-8406-dcf0b0f69082" />

```
 import pandas as pd
 from sklearn.feature_selection import SelectKBest, chi2, f_classif
 from sklearn.model_selection import train_test_split # Importing the missing function
 from sklearn.ensemble import RandomForestClassifier
 selected_features = ['age', 'maritalstatus', 'relationship', 'capitalgain', 'capitalloss',
 'hoursperweek']
 X = df[selected_features]
 y = df['SalStat']
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
 rf = RandomForestClassifier(n_estimators=100, random_state=42)
 rf.fit(X_train, y_train)
```
<img width="440" height="117" alt="image" src="https://github.com/user-attachments/assets/4360537c-807a-4f4e-88ba-828c083b945e" />

```
 y_pred = rf.predict(X_test)
 from sklearn.metrics import accuracy_score
 accuracy = accuracy_score(y_test, y_pred)
 print(f"Model accuracy using selected features: {accuracy}")
```

<img width="590" height="31" alt="Screenshot 2025-10-08 134817" src="https://github.com/user-attachments/assets/1895b904-dc45-4d8a-a74a-3ecfc9ca7e45" />

```
 !pip install skfeature-chappers
```
<img width="1355" height="512" alt="image" src="https://github.com/user-attachments/assets/e5f7dfdc-2223-4e05-bc78-50624d5e02a3" />

```
 import numpy as np
 import pandas as pd
 from skfeature.function.similarity_based import fisher_score
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.model_selection import train_test_split
 from sklearn.metrics import accuracy_score
```
```
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 df[categorical_columns] = df[categorical_columns].astype('category')
```
```
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 # @title
 df[categorical_columns]
```

<img width="917" height="496" alt="image" src="https://github.com/user-attachments/assets/c5653abc-ef65-4594-8dbf-515537849b3d" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
 k_anova = 5
 selector_anova = SelectKBest(score_func=f_classif,k=k_anova)
 X_anova = selector_anova.fit_transform(X, y)
 selected_features_anova = X.columns[selector_anova.get_support()]
 print("\nSelected features using ANOVA:")
 print(selected_features_anova)
```

<img width="809" height="60" alt="Screenshot 2025-10-08 135039" src="https://github.com/user-attachments/assets/fbf84049-79d4-4eef-83a2-5354271ee55a" />

```
 # Wrapper Method
 import pandas as pd
 from sklearn.feature_selection import RFE
 from sklearn.linear_model import LogisticRegression
 df=pd.read_csv("income(1) (1).csv")
 # List of categorical columns
 categorical_columns = [
 'JobType',
 'EdType',
 'maritalstatus',
 'occupation',
 'relationship',
 'race',
 'gender',
 'nativecountry'
 ]
 # Convert the categorical columns to category dtype
 df[categorical_columns] = df[categorical_columns].astype('category')
 df[categorical_columns] = df[categorical_columns].apply(lambda x: x.cat.codes)
 df[categorical_columns]
```

<img width="866" height="489" alt="image" src="https://github.com/user-attachments/assets/6485a22e-24b3-4e2f-ae3e-27e18d79b852" />

```
 X = df.drop(columns=['SalStat'])
 y = df['SalStat']
logreg = LogisticRegression()
n_features_to_select =6
rfe = RFE(estimator=logreg, n_features_to_select=n_features_to_select)
rfe.fit(X, y)
```

<img width="376" height="189" alt="image" src="https://github.com/user-attachments/assets/0144abc1-50db-493c-9745-20e254093c07" />





# RESULT:
Thus the program to read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file has been executed successfully. 
