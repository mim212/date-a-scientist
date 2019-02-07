#Import Libs
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import MultinomialNB
from scipy.stats import zscore
from sklearn.neighbors import KNeighborsRegressor

#Load data into dataframe and explore the column headings
df = pd.read_csv("profiles.csv")
# print(df.columns.values)

#Print Age distribution using histogram
# plt.hist(df.age, bins=20)
# plt.xlabel("Age")
# plt.ylabel("Frequency")
# plt.xlim(16, 80)
# plt.show()

#Print possible responses to zodiac sign
# print(df.sign.value_counts())

"""
Question - Can we predict the users sex with education level and income?

- K-Nearest Neighbors

Vs

- Naive Bayes Classifier

"""
#Map non-numerical data to a 'code' to represent different responses
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping, na_action='ignore')

education_mapping = {
"dropped out of high school": 0,
"working on high school": 1,
"graduated from high school": 2,
"high school": 3,
"dropped out of college/university": 4,
"dropped out of space camp": 5,
"dropped out of two-year college": 6,
"dropped out of masters program": 7,
"dropped out of ph.d program": 8,
"dropped out of law school": 9,
"dropped out of med school": 10,
"working on two-year college": 11,
"working on college/university": 12,
"working on masters program": 13,
"working on ph.d program": 14,
"working on space camp": 15,
"working on law school": 16,
"working on med school": 17,
"graduated from two-year college": 18,
"two-year college": 19,
"graduated from college/university": 20,
"college/university": 21,
"graduated from law school": 22,
"law school": 23,
"graduated from masters program": 24,
"masters program": 25,
"graduated from ph.d program": 26,
"ph.d program": 27,
"graduated from space camp": 28, 
"space camp": 29,
"graduated from med school": 30,
"med school": 31,
"No Response (NaN)": 32
}
df["education_code"] = df.education.map(education_mapping).fillna(32)

#Create list of education levels for plot labels
edu_list = []
for i in range(len(education_mapping)):
    edu_list.append(education_mapping.keys()[education_mapping.values().index(i)])

#Print the number of male and female responses
print(df.sex.value_counts())   

#Seperate education and income based on sex to test for correlation
edu_m = df.education_code[df['sex'] == 'm']
edu_f = df.education_code[df['sex'] == 'f']
inc_m = df.income[df['sex'] == 'm']
inc_f = df.income[df['sex'] == 'f']

#Plot income based on sex
plt.hist(inc_m, bins =50, color='blue')
plt.hist(inc_f, bins =50, color='red')
plt.xlabel("$ Income per year")
plt.ylabel("No of responses")
plt.xlim(0, 150000)
plt.legend(["Male", "Female"])
plt.show()

#Plot education based on sex
plt.hist(edu_m, bins =33, color='blue')
plt.hist(edu_f, bins =33, color='red')
ax = plt.subplot()
plt.xlabel("Level of education")
plt.ylabel("No of responses")
ax.set_xticks(range(len(edu_list)))
ax.set_xticklabels(edu_list)
plt.xticks(rotation=90)
plt.xlim(0, 32)
plt.legend(["Male", "Female"])
plt.show()

"""
The below function spans k-values and tests for accuracy. 
The most accurate k-value is then printed on the console.
Inputs: k_max = highest k value the function will test for
"""

def Accuracy_Span_KClassifier(k_max):   
    k_list = range(1,k_max)
    accuracies = []
    max_acc = 0
    max_acc_k = 0
    #Loop though all k-values up to k-max, listing the accuracy of each result
    for k in k_list:
        classifier = KNeighborsClassifier(n_neighbors = k)
        classifier.fit(training_data, training_labels)
        accuracy = classifier.score(validation_data, validation_labels)
        #Find max accuracy and record the k-value
        if accuracy > max_acc:
            max_acc = accuracy
            max_acc_k = k
        accuracies.append(accuracy)
    
    #Plot the results of the accuracy changing with k values
    plt.plot(k_list, accuracies, color='purple')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title("K Neighbors Accuracy")
    plt.show()
    print("Max accuracy found at k=%s" % max_acc_k)
    
#Normalise data using z-score normalisation to remove outliers (e.g. $1000000 income)
norm_income = zscore(df.income)
norm_edu = zscore(df.education_code)

#Offsetting z-scores to remove negitive values (issue with Naive Bayes Classifier)
norm_income = [x+1 for x in norm_income]
norm_edu = [x+3 for x in norm_edu]

#Combine the education level and salary into one array for testing
income_education = zip(norm_edu, norm_income)

#Split the sample data into training and validation sets (note: test size = 20%)
(training_data, validation_data, training_labels, validation_labels) = train_test_split(income_education, df.sex_code, test_size = 0.2, random_state = 42)

#Find Ideal k value (commented to reduce loading times)
# Accuracy_Span_KClassifier(50)

#Define K Neighbors Classifier (with previously found best k-value) and test accuracy
classifier_kneigh = KNeighborsClassifier(n_neighbors = 26)
classifier_kneigh.fit(training_data, training_labels)

print("Accuracy of K-Nearest Neighbors Classifier: %s" % classifier_kneigh.score(validation_data, validation_labels))

#Define Naive Bayes Classifier and test accuracy
classifier_nb = MultinomialNB()
classifier_nb.fit(training_data, training_labels)

print("Accuracy of Naive Bayes Classifier: %s" % classifier_nb.score(validation_data, validation_labels))


"""
# Question - Can we predict the users age with Income and job?

- Multiple Linear Regression 

Vs

- KNeighbors Regressor
"""
"""
The below function spans k-values and tests for accuracy. 
The most accurate k-value is then printed on the console.
Inputs: k_max = highest k value the function will test for
"""

def Accuracy_Span_KRegressor(k_max):   
    k_list = range(1,k_max)
    accuracies = []
    max_acc = 0
    max_acc_k = 0
    #Loop though all k-values up to k-max, listing the accuracy of each result
    for k in k_list:
        classifier = KNeighborsRegressor(k, weights="distance")
        classifier.fit(training_data, training_labels)
        accuracy = classifier.score(validation_data, validation_labels)
        #Find max accuracy and record the k-value
        if accuracy > max_acc:
            max_acc = accuracy
            max_acc_k = k
        accuracies.append(accuracy)
    
    #Plot the results of the accuracy changing with k values
    plt.plot(k_list, accuracies, color='purple')
    plt.xlabel('k')
    plt.ylabel('Validation Accuracy')
    plt.title("K Neighbors Regressor Accuracy")
    plt.show()
    print("Max accuracy found at k=%s" % max_acc_k)

#Map non-numerical data to a 'code' to represent different responses
job_mapping = {
"No Repsonse" : 1,
"other" : 2,
"student" : 3,
"science / tech / engineering" : 4,
"computer / hardware / software" : 5,
"artistic / musical / writer" : 6,
"sales / marketing / biz dev" : 7,
"medicine / health" : 8,
"education / academia" : 9,
"executive / management" : 10,
"banking / financial / real estate" : 11,
"entertainment / media" : 12,
"law / legal services" : 13,
"hospitality / travel" : 14,
"construction / craftsmanship" : 15,
"clerical / administrative" : 16,
"political / government" : 17,
"rather not say" : 18,
"transportation" : 19,
"unemployed" : 20,
"retired" : 21,
"military"  : 22
}
df["job_code"] = df.job.map(job_mapping).fillna(1)

#Normalise data using z-score normalisation to remove outliers
norm_age = zscore(df.age)

#Combine the job and salary info into one array for testing
income_job = zip(df.job_code, norm_income)

#Split the sample data into training and validation sets (note: test size = 20%)
(training_data, validation_data, training_labels, validation_labels) = train_test_split(income_job, norm_age, test_size = 0.2, random_state = 42)

#Define Multiple Linear Regressor
classifier_mlr = LinearRegression()
classifier_mlr.fit(training_data, training_labels)

#Print accuracy of MLR classifier
print("MLR training score: %s " % classifier_mlr.score(training_data, training_labels))
print("MLR validation score: %s " % classifier_mlr.score(validation_data, validation_labels))

#Find Ideal k value (commented to reduce loading times)
Accuracy_Span_KRegressor(100)

#Define SKNeighbors Regressor
classifier_kneighReg = KNeighborsRegressor(20, weights="distance")
classifier_kneighReg.fit(training_data, training_labels)

print("KNeighbors validation score: %s " % classifier_kneighReg.score(validation_data, validation_labels))
