import numpy as np
import pandas as pd
from sklearn.model_selection import KFold 

# kFold: Keeps track of how many folds we perform 
kFold = 10
# printOutput: List that stores the coefficient names which we will
# print out when we print the coefficients once testing is complete
printOutput = ["Cylinders", "Displacement", "Horsepower", "Weight", 
               "Acceleration", "Model Year", "Origin"]

'''
Description: Computes a linear regression model for a given training dataset
Parameters: Dataframes containing the training data features and labels
Return Type: Dataframe containing the intercept and coefficients for 
our linear regression model
'''
def regressionFormula(carFeatures, carLabels):
    # Here, we will use the optimized linear regression equation to 
    # train our model
    # Equation: b = (X^T * X)^-1 * X^T * y
    numSamples, numFeatures = carFeatures.shape
    carFeatures = pd.DataFrame(np.c_[np.ones(numSamples),carFeatures])
    return np.dot(np.dot(np.linalg.inv(np.dot(carFeatures.transpose(), 
        carFeatures)), carFeatures.transpose()), carLabels)

'''
Description: Validates our model using testing data to see how well our
model performs
Parameters: Dataframes containing the parameters for our model and the 
testing data's features/labels
Return Type: Int (Represents the RMSE for our model)
'''
def validate(coefficients, testFeatures, testLabels):
    # Here, we will test and validate how our model performs by finding
    # the RMSE for each fold
    # Equation: RMSE = sqrt(sum[i=1 to N] (yi - Xi * b)^2)
    numSamples, numFeatures = testFeatures.shape
    testFeatures = pd.DataFrame(np.c_[np.ones(numSamples),testFeatures])
    return (np.sum((np.dot(testFeatures, coefficients) - testLabels)**2)/numSamples)**0.5

'''
Description: Main function for this file
Parameters: None
Return Type: None
'''
def main():
    # First, read in the data and remove any samples that contain
    # missing data
    car_data = pd.read_fwf("auto-mpg.data", header=None, na_values="?")
    car_data = car_data.dropna()

    # Then, we will separate the features from the labels 
    # First, we get the labels and store them into a dataframe 
    # called mpg
    mpg = car_data.iloc[:,0]

    # Then, we will remove the labels from the data alongside the 
    # last column which contains the names of the cars. We will also 
    # convert all the data into floats, so we can perform our calculations
    # without type errors
    car_data = car_data.drop(0,axis=1)
    car_data = car_data.drop(8,axis=1)
    car_data = car_data.astype(float)
    
    # currFold: Keeps track of the current fold we are on
    currFold = 1
    
    # Now, we will find the folds we will be using to create our models
    # and we will proceed to loop through each fold
    fold = KFold(n_splits=kFold)

    for train, test in fold.split(car_data):
        # First, we will split our features into training/testing sets
        car_train = car_data.iloc[train,:]
        car_test = car_data.iloc[test,:]
        # Then, we will split our labels into training/testing sets
        mpg_train = mpg.iloc[train]
        mpg_test = mpg.iloc[test]

        # We will now call regressionFormula to get the parameters we 
        # will be using for our model (7 coefficients/1 intercept)
        coefficients = regressionFormula(car_train, mpg_train)

        # Afterwards, we will call validate to check how our model 
        # performs with the testing data
        RMSE = validate(coefficients, car_test, mpg_test)
        
        # Finally, print out the coefficients alongside the RMSE
        # for each fold to the user
        print(f"Fold {currFold}:")
        for i in range(1,len(coefficients)):
            print(f"\t{printOutput[i-1]} = {coefficients[i]:.5f}", end="\n")
        print(f"\n\tRMSE = {RMSE:.5f}")
        print("\n")
        currFold += 1

if __name__ == "__main__":
    main()
    