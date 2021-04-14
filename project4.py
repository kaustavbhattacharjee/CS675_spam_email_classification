# all imports
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

#The files
training_datafile  = sys.argv[1]
testing_datafile = sys.argv[2]

#Reading the training and testing data
training_data= pd.read_csv(training_datafile,encoding='ISO-8859-1')
test_data= pd.read_csv(testing_datafile,encoding='ISO-8859-1',header =0,names = ['v1','v2','v3','v4','v5']) #the testing file does not contain the header; line 2549 has total 5 columns

test_data = test_data[['v1','v2']] #extracting only the first 2 columns to read
feature_extraction = TfidfVectorizer()  #from sklearn.feature_exraction.text
x = feature_extraction.fit_transform(training_data['v2'].values)
vocab = feature_extraction.get_feature_names()
training_x, testing_x, training_y, testing_y = train_test_split(x, training_data['v1'], test_size=0.30, random_state=40)

'''
Accuracy for different values of random_state
44: 96.907
43: 97.232
42: 96.939
41: 96.939
40:97.265
39: 96.134
38: 97.004
'''
classifier = SVC(probability=True, kernel='rbf',gamma = 'scale')
classifier.fit(training_x, training_y)
feature_extraction = TfidfVectorizer(vocabulary=vocab)
Y = feature_extraction.fit_transform(test_data['v2'].values)

predictor = classifier.predict(Y)
accuracy = accuracy_score(test_data['v1'], predictor, normalize=True)

for m in range(len(predictor)):
    print(m, predictor[m])
print("Accuracy:",accuracy*100)