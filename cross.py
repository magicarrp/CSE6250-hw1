import models_partc
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from numpy import mean


import utils

# USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

# USE THIS RANDOM STATE FOR ALL OF YOUR CROSS VALIDATION TESTS, OR THE TESTS WILL NEVER PASS
RANDOM_STATE = 545510477

#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_kfold(X,Y,k=5):
	#TODO:First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the folds
    
    accuracy_list = []
    auc_list = []
    kf = KFold(n_splits=k)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, y_train)
        
        Y_pred = logisticRegr.predict(X_test)
        accuracy_list.append(accuracy_score(y_test, Y_pred))
        auc_list.append(roc_auc_score(y_test, Y_pred))
    return sum(accuracy_list)/k,sum(auc_list)/k


#input: training data and corresponding labels
#output: accuracy, auc
def get_acc_auc_randomisedCV(X,Y,iterNo=5,test_percent=0.2):
	#TODO: First get the train indices and test indices for each iteration
	#Then train the classifier accordingly
	#Report the mean accuracy and mean auc of all the iterations
	
    accuracy_list = []
    auc_list = []
    kf = ShuffleSplit(n_splits=iterNo,test_size=test_percent)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        
        logisticRegr = LogisticRegression()
        logisticRegr.fit(X_train, y_train)
        
        Y_pred = logisticRegr.predict(X_test)
        accuracy_list.append(accuracy_score(y_test, Y_pred))
        auc_list.append(roc_auc_score(y_test, Y_pred))
    return sum(accuracy_list)/iterNo,sum(auc_list)/iterNo


def main():
	X,Y = utils.get_data_from_svmlight("../deliverables/features_svmlight.train")
    # X,Y = load_svmlight_file("C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/features_svmlight.train")
	print("Classifier: Logistic Regression__________")
	acc_k,auc_k = get_acc_auc_kfold(X,Y)
	print(("Average Accuracy in KFold CV: "+str(acc_k)))
	print(("Average AUC in KFold CV: "+str(auc_k)))
	acc_r,auc_r = get_acc_auc_randomisedCV(X,Y)
	print(("Average Accuracy in Randomised CV: "+str(acc_r)))
	print(("Average AUC in Randomised CV: "+str(auc_r)))

if __name__ == "__main__":
	main()

