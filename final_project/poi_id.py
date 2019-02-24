#!/usr/bin/python

from __future__ import division
import sys
import pickle
from matplotlib import pyplot as plt
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".


features_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
                 'exercised_stock_options', 'expenses', 'from_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 
                 'long_term_incentive', 'other', 'restricted_stock', 
                 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 
                 'to_messages', 'total_payments', 'total_stock_value']


"""
                ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
                 'exercised_stock_options', 'expenses', 'from_messages', 
                 'from_poi_to_this_person', 'from_this_person_to_poi', 'loan_advances', 
                 'long_term_incentive', 'other', 'restricted_stock', 
                 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 
                 'to_messages', 'total_payments', 'total_stock_value']
"""

### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
"""
First, I would like to convert the data into dataframe for easy data wrangling and analsis.
I would be using pandas.DataFrame.from_records to convert structured data into dataframe.
"""

my_df = pd.DataFrame.from_records(list(data_dict.values()))
print my_df


"""
Let's take a look at some characteristics of the dataframe.
"""

print "Number of data points: ", len(my_df)

print "Number of features: ", len(my_df.columns)

poi_total = my_df.groupby('poi').size()
print "Total POIs : ", poi_total.iloc[1]
print "Total Non-POIs: ", poi_total.iloc[0]


"""
After printing my dataframe, it looked messy and unstructured to me.
So, I decided to use insiders as an index.
"""

insiders = pd.Series(list(data_dict.keys()))
my_df.set_index(insiders, inplace=True)
print list(my_df)
print my_df.head()

"""
This looks much better.
Next, I would like to see the column datatypes.
"""

print my_df.dtypes

"""
I would like to change the datatypes into floats and change NaN to 0 for easy analysis
"""

my_df_update1 = my_df.apply(lambda x: pd.to_numeric(x, errors = 'coerce')).copy().fillna(0)
print my_df_update1.dtypes
print my_df_update1.head()

"""
This looks good.
I think that email_address adds no value in my analysis.
So, I would remove the email_address column.
"""

my_df_update1.drop('email_address', inplace = True, axis = 1)
print list(my_df_update1)
print my_df_update1.head()

"""
Perfect, looks like I am done with task 1. Moving on to task 2.
"""

### Task 2: Remove outliers
    

"""
Let's plot bonus and salary plot to see if anything stands out
"""

plt.scatter(my_df_update1['salary'],
            my_df_update1['bonus'])
            
plt.xlabel("Salary")
plt.ylabel("Bonus")
plt.show()


"""
Ok, looks like there is one value that is way too big. Let's find out who is that insider.
"""

print "Insider with largest salary: ", my_df_update1['salary'].idxmax()

 
#Ok, so first outlier is "TOTAL".

"""
Next, I want to see all the insider names, may be I will catch something.
"""

for i in range(len(my_df_update1)):
    print "Insider name: ", my_df_update1.index[i]

"""    
The name "THE TRAVEL AGENCY IN THE PARK" looks weird. 
This does did not look like a typo to me.
I did a google search and found that it was a travel agency co-owned by Ken Lay's sister Sharon Lay. 
(source: http://content.time.com/time/magazine/article/0,9171,198885,00.html).
I read about Enron's history and watched documentaries during working on mini projects.
So, I was aware of Ken Lay. I will remove "THE TRAVEL AGENCY IN THE PARK" as the outlier,
but this does prove that Ken Lay is clearly a point of interest and linked to fraud.
"""

# "Second outlier is "THE TRAVEL AGENCY IN THE PARK".


"""
Lastly, I would like to see if there is anyone with all values missing.
For this, I would like to print insiders with missing total_payments and total_stock_value.
"""

missing_values = my_df_update1[(my_df_update1['total_payments'] == 0) & 
                     (my_df_update1['total_stock_value'] == 0)]
print "Insiders with missing values: \n", missing_values


"""
Ok, looks like Eugene Lockhart is the only one with all the missing values and 
also he is non-poi.
"""

# Third outlier is "EUGENE LOCKHART".


"""
Removing all 3 outliers.
"""

my_df_update1.drop('LOCKHART EUGENE E', inplace = True)
my_df_update1.drop('THE TRAVEL AGENCY IN THE PARK', inplace = True)
my_df_update1.drop('TOTAL', inplace = True)

# Checking to make sure outliers are dropped
for i in range(len(my_df_update1)):
    print "Insider name: ", my_df_update1.index[i]


"""
Perfect. Moving on to task 3.
"""

### Task 3: Create new feature(s)

"""
I am pretty sure that POIs left digital trace while communicating through emails.
By exploring their email communication, I am sure we can get some sort of idea.
I would like to create  2 features with from/to emails.
"""

my_df_update1['email_ratio_from_poi'] = my_df_update1['from_poi_to_this_person'] / my_df_update1['from_messages']
my_df_update1['email_ratio_to_poi'] = my_df_update1['from_this_person_to_poi'] / my_df_update1['to_messages']


plt.scatter(my_df_update1['email_ratio_from_poi'][my_df_update1['poi'] == True],
            my_df_update1['email_ratio_to_poi'][my_df_update1['poi'] == True],
            color = 'red', label = 'POI')


plt.scatter(my_df_update1['email_ratio_from_poi'][my_df_update1['poi'] == False],
            my_df_update1['email_ratio_to_poi'][my_df_update1['poi'] == False],
            color = 'green', label = 'Non-POI')

plt.xlabel("Emails From POI")
plt.ylabel("Emails To POI")
plt.show()

"""
The plot shows that POIs did send large amount of emails.
"""

"""
Next, let's clean infinity values if there are any.
"""

my_df_update1 = my_df_update1.replace('inf', 0)
my_df_update1 = my_df_update1.fillna(0)

print list(my_df_update1)

"""
Now I am ready to convert my dataframe into dictionary as requested by Udacity.
"""

enron_dict = my_df_update1.to_dict('index')


### Store to my_dataset for easy export below.
my_dataset = data_dict

"""
Final feature list.
"""

feature_list = ['poi', 'bonus', 'deferral_payments', 'deferred_income', 'director_fees', 
                'exercised_stock_options', 'expenses', 'from_messages', 'from_poi_to_this_person', 
                'from_this_person_to_poi', 'loan_advances', 'long_term_incentive', 'other', 
                'restricted_stock', 'restricted_stock_deferred', 'salary', 'shared_receipt_with_poi', 
                'to_messages', 'total_payments', 'total_stock_value']


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Features for pipeline
select = SelectKBest()
scaler = MinMaxScaler()

# Classifiers I will try
knn = KNeighborsClassifier()
dec_tree = DecisionTreeClassifier()

#Pipeline for KNeighborsClassifier
#pipeline = Pipeline(steps=[('feature_selection', select),('scaler', scaler),('knn', knn)])

#Pipeline for DecisionTreeClassifier
pipeline = Pipeline(steps=[('feature_selection', select),("dec_tree",dec_tree)])


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
#from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
#features_train, features_test, labels_train, labels_test = \
 #   train_test_split(features, labels, test_size=0.3, random_state=42)


"""
#First classifier: k-nearest neighbors

# Parameters for grid search
parameters = dict(feature_selection__k=[1,2,3,4,5,6],
                  knn__n_neighbors=[1],
                  knn__leaf_size=[1],
                  knn__algorithm=['auto'])

sss = StratifiedShuffleSplit(n_splits = 10,test_size = 0.30,random_state = 42)

grid_search = GridSearchCV(pipeline, param_grid=parameters, scoring="f1", cv=sss, error_score=0)

grid_search.fit(features_train, labels_train)

predictions = grid_search.predict(features_test)

clf = grid_search.best_estimator_

print "\n", "Best parameters are: ", grid_search.best_params_, "\n"

print "Precision for KNeighborsClassifier: ", precision_score(predictions, labels_test)
print "Recall for KNeighborsClassifier: ", recall_score(predictions, labels_test)
"""

#Second classifier: DecisionTreeClassifier

# Parameters for grid search
parameters = dict(feature_selection__k=[3,4,5,6,7,8,9,10,11,12,13,14,15], 
                  dec_tree__criterion = ['gini', 'entropy'],
                  dec_tree__max_depth = [None, 1, 2, 3, 4],
                  dec_tree__min_samples_split = [2,4,6,10],
                  dec_tree__class_weight = ['balanced'],
                  dec_tree__random_state = [42])

sss = StratifiedShuffleSplit(n_splits = 10, test_size = 0.30, random_state = 42)

grid_search = GridSearchCV(pipeline, param_grid=parameters, scoring="f1", cv=sss, error_score=0)

grid_search.fit(features_train, labels_train)

predictions = grid_search.predict(features_test)

clf = grid_search.best_estimator_

print "Precision for DecisionTreeClassifier: ", precision_score(predictions, labels_test)
print "Recall for DecisionTreeClassifier: ", recall_score(predictions, labels_test)



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)


# Printing list of features and scores

features_bool = grid_search.best_estimator_.named_steps['feature_selection'].get_support()

features_selected = [a for a , b in zip(features_list[1:],
                                        features_bool) if b]

feature_score =  grid_search.best_estimator_.named_steps['feature_selection'].scores_

feature_selected_scores = feature_score[features_bool]

feature_score_list = pd.DataFrame({'Feature':features_selected, 'Score':feature_selected_scores})

print feature_score_list

