
# import libaries
import numpy as py
import pandas as pd
import pprint

# import datasets
data_set = pd.read_csv('../datasets/Data.csv')

X = data_set.iloc[: , :-1].values
# pprint.pprint(X)
Y = data_set.iloc[: , 3].values

# Handling the missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0) # The axis along which to impute.
imputer = imputer.fit(X[:,1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
# pprint.pprint(X)

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
# pprint.pprint(X)


onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
pprint.pprint(X)

labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
# pprint.pprint(Y)

# Splitting the datasets into training sets and Test sets
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)