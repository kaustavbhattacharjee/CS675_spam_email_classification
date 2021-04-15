# Spam Email Classification

## Objective:
Write a Python program that learns a TFIDF model from spam_train.csv. Use 
the model to get TFIDF vectors for spam_test.csv. 

Your program will then learn an SVM model (LinearSVC in sklearn) on the 
TFIDF of spam_train.csv and predict spam or ham for spam_test.csv.

Your model should achieve at least 95% accuracy to get full points.

Your program takes in two files spam_train.csv and spam_test.csv and 
outputs the predicted labels as well as their accuracy on the spam_test.csv 
file.

Directories: /afs/cad/courses/ccs/s20/cs/675/002/<ucid>.
For example if your ucid is abc12 then copy your programs into
/afs/cad/courses/ccs/s20/cs/675/002/abc12.

Your completed program is due before 1pm May 11th 2020

## How to Execute Program

Run the following command
```
python3 project4.py spam_train.csv spam_test.csv
```

## Proof of Submission
![Photo](Proof_of_Submission_Project4.png)