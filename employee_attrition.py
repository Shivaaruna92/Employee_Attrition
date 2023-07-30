#Importing the packages.
#Data processing packages.
import numpy as np 
import pandas as pd 

#Visualization packages.
import matplotlib.pyplot as plt 
import seaborn as sns 

#Machine Learning packages.
from sklearn.svm import SVC,NuSVC
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler    
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from google.colab import files

#Suppress warnings.
import warnings
warnings.filterwarnings('ignore')
#Load the Data.
uploaded =  files.upload()
#Import Employee Attrition data.
print("Data Set that's been uploaded.\n")
data=pd.read_csv('attrition_dataset.csv')
data.head()
#Print the column names
print("Variables in the data set")
i=1
for col in data.columns:
    print(" ",i," - ",col)
    i+=1
#No of Rows and Columns.
print("No of rows and no of columns in the data set")
data.shape
#Checking Data for any null values.
if data.isnull().values.any():
    print("There are null values in the data set so it is returning",end=" ")
else:
    print("There are no null values in the data set so it is returning",end=" ")
data.isnull().values.any()
#Print all of the Data Types and their Unique Values.
for column in data.columns:
    if data[column].dtype == object:
        unique_values = data[column].unique()
        value_counts = data[column].value_counts()
        print(str(column) + ':\n')
        print(tabulate({'Unique Values': unique_values, 'Value Counts': value_counts}, headers='keys', tablefmt='psql'))
        print('____________________________\n')
#Showing no of Employees Stay and Left by Age.
plt.subplots(figsize = (15 , 10))
sns.countplot(x = 'Age', hue = 'Attrition', data = data, palette = 'colorblind')
#Describing the data for Some Statistics.
data.describe().T
#Calculate the correlation matrix
corr_matrix = data.corr()
print("Correlation matrix is calculated")
#Create a mask to hide the upper triangle of the correlation matrix
mask = np.zeros_like(corr_matrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
print("Mask for upper triangle is created for the correlation matrix")
#Plot the heatmap of the correlation matrix
plt.figure(figsize=(30,25))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', xticklabels=corr_matrix.columns.values, yticklabels=corr_matrix.columns.values, annot_kws={"size": 16}, fmt=".2f", linewidths=0.5, square=True)
plt.xticks(rotation=45, ha='right', fontsize=18)
plt.yticks(fontsize=18)
plt.title("Correlation Matrix", fontsize=40)
plt.show()
#Remove highly correlated features
threshold = 0.7
corr_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
print("Highly correlated features have been segregated from the data set")
#Drop highly correlated features
data.drop(corr_features, axis=1, inplace=True)
print("Segregated features were removed")
#No of Rows and Columns.
print("No of rows and no of columns in the data set after removing highly correlated features")
data.shape
#Print the column names
print("Variables now in the data set")
i=1
for col in data.columns:
    print(" ",i," - ",col)
    i+=1
#Displaying variables that will be used further after dropping some variables.
data.head()
#Count of No of Employees Stay and Left.
print("Checking how many employees will leave and stay in the company based on the real time data set")
print("--> No - employee will STAY in the company")
print("--> Yes - employee will LEAVE the company")
print("---------------------------------------------------------------------------------------------")
data['Attrition'].value_counts()
#Visualizing the Employees Stay and Left.
plt.subplots(figsize=(9,6))
sns.countplot(x='Attrition', data=data)
plt.xlabel('Employment Status')
plt.ylabel('Number of Employees')
plt.title('Number of Employees Stayed and Left')
plt.legend(labels=['Stayed', 'Left'])
plt.show()
#Convert Categorical values to Numeric Values.
#A lambda function is a small anonymous function.
#A lambda function can take any number of arguments, but can only have one expression.
print("Data Set after converting the target field into numerical values.\n")
data['Attrition']=data['Attrition'].apply(lambda x : 1 if x=='Yes' else 0)
data.head()
#This function is used to convert Categorical values to Numerical values
print("Convertin Categorical values to Numerical values.\n")
data=pd.get_dummies(data)
data.head()
#Separating Feature and Target matrices
X = data.drop(['Attrition'], axis=1)
y=data['Attrition']


#Feature scaling is a method used to standardize the range of independent variables or features of data.
#Since the range of values of raw data varies widely, in some machine learning algorithms, objective functions will not work properly without normalization. 
scale = StandardScaler()
X = scale.fit_transform(X)


# Split the data into Training set and Testing set
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size =0.2,random_state=42)


#Function to Train and Test Machine Learning Model
def train_test_ml_model(X_train,y_train,X_test,Model):
    model.fit(X_train,y_train) #Train the Model
    y_pred = model.predict(X_test) #Use the Model for prediction

    # Test the Model
    cm = confusion_matrix(y_test,y_pred)
    accuracy = round(100*np.trace(cm)/np.sum(cm),1)

    #Plot/Display the results
    cm_plot(cm,Model)
    temp="\n\n                  Accuracy of the Model " +Model+" "+str(accuracy)+"%"
    print(temp)


#Function to plot Confusion Matrix
def cm_plot(cm,Model):
    plt.clf()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
    classNames = ['Negative','Positive']
    plt.title('Comparison of Prediction Result for '+ Model)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    tick_marks = np.arange(len(classNames))
    plt.xticks(tick_marks, classNames, rotation=45)
    plt.yticks(tick_marks, classNames)
    s = [['TN','FP'], ['FN', 'TP']]
    for i in range(2):
        for j in range(2):
            plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
    plt.show()
#Perform Predictions using Machine Learning Algorithms
#svm
Model = "SVC"
#Create the Model
model=SVC()
train_test_ml_model(X_train,y_train,X_test,Model)
#svm
Model = "NuSVC"
#Create the Model
model=NuSVC(nu=0.285)                                       
train_test_ml_model(X_train,y_train,X_test,Model)
#xgboost
Model = "XGBClassifier"
#Create the Model
model=XGBClassifier() 
train_test_ml_model(X_train,y_train,X_test,Model)
#neighbors
Model = "KNeighborsClassifier"
#Create the Model
model=KNeighborsClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#naive_bayes
Model = "GaussianNB"
#Create the Model
model=GaussianNB()
train_test_ml_model(X_train,y_train,X_test,Model)
#linear_model
Model = "SGDClassifier"
#Create the Model
model=SGDClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#linear_model
Model = "LogisticRegression"
#Create the Model
model=LogisticRegression()
train_test_ml_model(X_train,y_train,X_test,Model)
#tree
Model = "DecisionTreeClassifier"
#Create the Model
model=DecisionTreeClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#tree
Model = "ExtraTreeClassifier"
#Create the Model
model=ExtraTreeClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#discriminant_analysis
Model = "QuadraticDiscriminantAnalysis"
#Create the Model
model = QuadraticDiscriminantAnalysis()
train_test_ml_model(X_train,y_train,X_test,Model)
#discriminant_analysis
Model = "LinearDiscriminantAnalysis"
#Create the Model
model=LinearDiscriminantAnalysis()
train_test_ml_model(X_train,y_train,X_test,Model)
#ensemble
Model = "RandomForestClassifier"
#Create the Model
model=RandomForestClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#ensemble
Model = "AdaBoostClassifier"
#Create the Model
model=AdaBoostClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
#ensemble
Model = "GradientBoostingClassifier"
#Create the Model
model=GradientBoostingClassifier()
train_test_ml_model(X_train,y_train,X_test,Model)
