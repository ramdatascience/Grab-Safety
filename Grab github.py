
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


# In[4]:


#read in the first set of training data.
data0=pd.read_csv('G:/Grab/safety/part0.csv')
Labels=pd.read_csv('G:/Grab/safety/labels.csv')
data0.head()


# In[5]:


Labels.head()


# In[6]:


data0.describe()


# In[7]:


#checking if there are any null values in the data0 dataset.
#if so then drop these rows of data, as they could affect the performance of the model training

if (data0.isnull().sum().sum()!=0):
    data0.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
    


# In[8]:


#checking if the Labels dataset have all unique BookingIDs.
#If there are duplicated BookingIDs, then drop these rows
#as they will confuse training of the classification algo.
if ((len(Labels))!=(len(Labels['bookingID'].unique()))):
    duplicateRows_Labels=Labels[Labels.duplicated(['bookingID'],keep=False)]
    for x in range(len(duplicateRows_Labels)):
        Labels.drop(duplicateRows_Labels.index[x],inplace=True)


# In[9]:


#confirm that these duplicated erroneous rows in Labels dataset have been dropped
duplicateRows_Labels=Labels[Labels.duplicated(['bookingID'],keep=False)]


# In[10]:


#take a peek to ensure no more duplicate labels
duplicateRows_Labels


# In[11]:


#we merge the Labels dataset into the Features dataset, using bookingID as the key to merge both datasets
Combined_Dataset=pd.merge(data0, Labels, on='bookingID',
         left_index=True, right_index=False, sort=False)


# In[12]:


#show the column headings in the merged dataset
Combined_Dataset.columns


# In[13]:


#take a peek at the Combined Dataset
Combined_Dataset.head()


# In[14]:


#We split the combined dataset into its constituent independent and dependent variables
#BookingID is deemed not relevant in the safety analysis and is not included as indepdendent variables list
#iv=Combined_Dataset[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']]
iv = Combined_Dataset.drop(['bookingID','label'],axis=1)
dv=Combined_Dataset[['label']]
iv.columns


# In[15]:


dv.columns


# In[16]:


iv.head()


# In[17]:


#Perform feature scaling to normalise all variabls to comparable scales so that 
#the analysis will not be skewed by certain variables taking on large values.

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']] = sc.fit_transform(iv[['Accuracy', 'Bearing', 'acceleration_x', 'acceleration_y','acceleration_z', 'gyro_x', 'gyro_y', 'gyro_z', 'second', 'Speed']])


# In[17]:


iv.head()


# In[18]:


#Use Logistic Regression classification technique
#Apply Recursive Feature Elimination (RFE) method for automatic feature selection to remove unimportant features
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

log_reg=LogisticRegression(random_state=1)
#log_reg.fit(iv_train,dv_train)
#log_reg.predict(iv_test)

# create the RFE model for the svm classifier 
# and select attributes
rfe = RFE(log_reg, 5)
rfe = rfe.fit(iv, dv)

# print summaries for the selection of attributes
print(rfe.support_)
print(rfe.ranking_)


# In[19]:


idx=iv.columns
idx


# In[20]:


#reduced list of features should be
#'Accuracy', 'Bearing', 'acceleration_y', 'acceleration_z', 'Speed'
reduced_features=[]
reduced_features_withKey=['bookingID']

for i in range(len(rfe.ranking_)):
    if (rfe.ranking_[i]==1):
        reduced_features.append(idx[i])
        reduced_features_withKey.append(idx[i])

reduced_features_withKey


# In[21]:


#Now that we have completed features selection, we will create the full training dataset
#using the reduced list of features

#read in the training data subsets based on reduced features.
data0=pd.read_csv('G:/Grab/safety/part0.csv',usecols=reduced_features_withKey)
data1=pd.read_csv('G:/Grab/safety/part1.csv',usecols=reduced_features_withKey)
data2=pd.read_csv('G:/Grab/safety/part2.csv',usecols=reduced_features_withKey)
data3=pd.read_csv('G:/Grab/safety/part3.csv',usecols=reduced_features_withKey)
data4=pd.read_csv('G:/Grab/safety/part4.csv',usecols=reduced_features_withKey)
data5=pd.read_csv('G:/Grab/safety/part5.csv',usecols=reduced_features_withKey)
data6=pd.read_csv('G:/Grab/safety/part6.csv',usecols=reduced_features_withKey)
data7=pd.read_csv('G:/Grab/safety/part7.csv',usecols=reduced_features_withKey)
data8=pd.read_csv('G:/Grab/safety/part8.csv',usecols=reduced_features_withKey)
data9=pd.read_csv('G:/Grab/safety/part9.csv',usecols=reduced_features_withKey)


# In[22]:


ReducedFeatures_Dataset=pd.concat([data0,data1,data2,data3,data4,data5,data6,data7,data8,data9],axis=0)
len(ReducedFeatures_Dataset)


# In[23]:


#checking  to see if there are any null values in the df_data0 dataset.
#if so then drop these rows of data, as they could affect the performance of the model training
if (ReducedFeatures_Dataset.isnull().sum().sum()!=0):
    educedFeatures_Dataset.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)


# In[25]:


ReducedFeatures_Dataset.head()


# In[26]:


len(ReducedFeatures_Dataset)


# In[27]:


#we merge the Labels dataset into the reduced features dataset, using bookingID as the key to merge both datasets
df_Combined_Dataset=pd.merge(ReducedFeatures_Dataset, Labels, on='bookingID',
         left_index=True, right_index=False, sort=False)


# In[30]:


#show the column headings in the merged dataset
df_Combined_Dataset.columns


# In[28]:


df_Combined_Dataset.head()


# In[29]:


#We split the combined dataset into its constituent independent and dependent variables
#BookingID is deemed not relevant in the safety analysis and is not included as indepdendent variables list
iv=df_Combined_Dataset[reduced_features]
dv=df_Combined_Dataset[['label']]


# In[33]:


#take a peek at the iv dataset BEFORE feature scaling
iv.head()


# In[30]:


#Perform feature scaling to normalise all variabls to comparable scales so that 
#the analysis will not be skewed by certain variables taking on large values.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
iv[reduced_features] = sc.fit_transform(iv[reduced_features])


# In[35]:


#take a peek at the iv dataset AFTER feature scaling
iv.head()


# In[31]:


from sklearn.model_selection import train_test_split

df1=df_Combined_Dataset['bookingID'] #extract the bookingID column into a temporary dataframe
iv_withBookingID=pd.concat([df1,iv], axis=1) #concatenate the bookingID into the iv set called iv_withBookingID

#split the iv set into training and test data with 80/20 split
iv_train_withBookingID,iv_test_withBookingID,dv_train,dv_test=train_test_split(iv_withBookingID,dv,test_size=0.2,random_state=0)


# In[93]:


#take a peek at the iv training set (with bookingID column)
iv_train_withBookingID.head()


# In[94]:


#take a peek at the iv test set (with bookingID column)
iv_test_withBookingID.head()


# In[32]:


#extract the bookingID from iv train and iv test sets,
#in preparation for sending to classification algo for training / prediction
iv_train=iv_train_withBookingID[reduced_features]
iv_test=iv_test_withBookingID[reduced_features]


# In[96]:


#take a peek at the iv train set, to ensure bookingID column has been successfully removed.
iv_train.head()


# In[97]:


#take a peek at the iv test set, to ensure bookingID column has been successfully removed.
iv_test.head()


# In[33]:


#Perform Logistic Regression classification technique
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression(random_state=1)
log_reg.fit(iv_train,dv_train)
dv_predict=log_reg.predict(iv_test)


# In[ ]:


#check the accuracy score for Logistic regression
accuracy_score(dv_test, dv_predict)


# In[ ]:


#confusion matrix for Logistic regression
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dv_test, dv_predict)


# In[ ]:


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(iv_train,dv_train)
dv_predict = classifier.predict(iv_test)


# In[ ]:


#check the accuracy score for SVM Model
accuracy_score(dv_test, dv_predict)


# In[ ]:


# Making the Confusion Matrix for SVM model

cm = confusion_matrix(dv_test, dv_predict)


# In[ ]:


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = iv_train, y = dv_train, cv = 10)
accuracies.mean()
accuracies.std()


# In[ ]:


# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV
parameters = [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
              {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]}]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(iv_train, dv_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_


# In[ ]:


#finally trying with advanced XGBOOST algorithm
# Fitting XGBoost to the Training set
from xgboost import xgb
classifier = xgb()
classifier.fit(iv_train,dv_train)

# Predicting the Test set results
y_pred = classifier.predict(iv_test)



# In[ ]:


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(dv_test, dv_predict)

