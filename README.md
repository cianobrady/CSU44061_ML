# CSU44061_ML

The model I have created for individual competiton is a linear regression model in python.

# Reading Files

The training and test files are read in using the pandas library. The test file's "Income" column is dropped as this is a different name to the equivalent column in tne training file. The files are then appended so they both the dataset of both can be edited easily.

# Data

The numerical columns I used were 'Year of Record', 'Age', 'Size of City', 'Wears Glasses' and 'Body Height [cm]'.

The categorical columns I used were 'Country', 'University Degree' and 'Gender'. This data was handled using One Hot Encoding.

# Training Data

I used scikit learn as my Linear Regression library. I trained the model with the training data and was able to predict the data using this model. I then outputted to a new csv file named out.csv
