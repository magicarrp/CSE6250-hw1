import utils
import pandas as pd
from sklearn.datasets import load_svmlight_file
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
#Note: You can reuse code that you wrote in etl.py and models.py and cross.py over here. It might help.
# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def aggregate_events(filtered_events_df, feature_map_df):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 c

    Suggested steps:
    1. Replace event_id's with index available in event_feature_map.csv
    2. Remove events with n/a values
    3. Aggregate events using sum and count to calculate feature value
    4. Normalize the values obtained above using min-max normalization(the min value will be 0 in all scenarios)
    
    
    IMPORTANT:
    Save aggregated_events to a csv file in the deliverables folder named as etl_aggregated_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header .
    For example if you are using Pandas, you could write: 
        aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)

    Return filtered_events
    '''
    # filtered_events_df = filtered_events
    # feature_map_df = feature_map
    filtered_events_df.dropna(subset=['value'], inplace=True)
    
    # merge1['len'] = merge1['event_id'].str.len()
    # merge1.groupby(['len']).count()
    # merge1[merge1['len'] == 11].head(10)
    
    
    filtered_events_df['event_id'] =  filtered_events_df['event_id'].str.strip()
    agg1 = filtered_events_df.groupby(['patient_id','event_id']).agg({'value':['sum','count']})
    agg1 = agg1['value']
    agg1.reset_index(inplace=True)
    # agg1.columns
    
    agg2 = agg1.groupby(['event_id'])[['sum','count']].max()
    agg3 = pd.merge(agg1,agg2,how='left',on='event_id',suffixes=('','_max'))
    
    agg3['sum'] = agg3['sum']/agg3['sum_max']
    agg3['count'] = agg3['count']/agg3['count_max']
    
    def f(x):
        start4 = x.event_id[:4]
        if start4=='DIAG' or start4=='DRUG':
            return x['sum']
        else:
            return x['count']
        
    agg3['feature_value'] = agg3.apply(f, axis=1)
    
    aggregated_events = pd.merge(agg3,feature_map_df,on='event_id',how='left')
    aggregated_events.rename(columns = {'idx':'feature_id'},inplace=True)
    aggregated_events = aggregated_events[['patient_id', 'feature_id', 'feature_value']]
    # events[events['patient_id']==1053]
    # aggregated_events[aggregated_events['patient_id']==1053]
    
    return aggregated_events

def create_features(events, feature_map):
    
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(events,feature_map)

    '''
    TODO: Complete the code below by creating two dictionaries - 
    1. patient_features :  Key - patient_id and value is array of tuples(feature_id, feature_value)
    2. mortality : Key - patient_id and value is mortality label
    '''
    
    key = aggregated_events['patient_id'][0]
    patient_features = {}
    tuples = []
    for row in aggregated_events.itertuples():
        #print((row.feature_id, row.feature_value))
        
        if row.patient_id==key:
            tuples.append((row.feature_id, row.feature_value))

        else:           
            patient_features[key] = tuples
            key = row.patient_id
            tuples = []
            tuples.append((row.feature_id, row.feature_value))        
        
    # merge2 = pd.merge(filtered_events,mortality,on='patient_id',how='left')
    # merge2.fillna(value = {'label':0}, inplace=True)
    # mortality = dict(zip(merge2['patient_id'],merge2['label']))

    return patient_features

def save_svmlight(patient_features):
    
    '''
    TODO: This function needs to be completed

    Refer to instructions in Q3 d

    Create two files:
    1. op_file - which saves the features in svmlight format. (See instructions in Q3d for detailed explanation)
    2. op_deliverable - which saves the features in following format:
       patient_id1 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...
       patient_id2 label feature_id:feature_value feature_id:feature_value feature_id:feature_value ...  
    
    Note: Please make sure the features are ordered in ascending order, and patients are stored in ascending order as well.     
    '''
    # deliverable1 = open(op_file, 'wb')
    # deliverable2 = open(op_deliverable, 'wb')
    test_features = open('C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.txt','w+')
    test_svm = open('C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.train', 'wb')
    
    
    # test = sorted(patient_features[41], key = lambda x: x[0])
    
    # out1 = ""
    # for i in patient_features:
    #     out1 = out1 + str(mortality[i]) + ' '
    #     for j in sorted(patient_features[i], key = lambda x: x[0]):
    #         out1 = out1 + str(int(j[0])) + ':'+ str(format(j[1], '.6f')) + ' '
    #     out1 = out1 + '\n'
        
    out2 = ""
    for i in patient_features:
        out2 = out2 + str(int(i)) + ' '
        for j in sorted(patient_features[i], key = lambda x: x[0]):
            out2 = out2 + str(int(j[0])) + ':'+ str(format(j[1], '.6f')) + ' '
        out2 = out2 + '\n'
        
    #print(out2)
    # deliverable1.write(bytes(out1,'UTF-8')); #Use 'UTF-8'
    test_features.write(out2)
    test_features.close()

'''
You may generate your own features over here.
Note that for the test data, all events are already filtered such that they fall in the observation window of their respective patients. Thus, if you were to generate features similar to those you constructed in code/etl.py for the test data, all you have to do is aggregate events for each patient.
IMPORTANT: Store your test data features in a file called "test_features.txt" where each line has the
patient_id followed by a space and the corresponding feature in sparse format.
Eg of a line:
60 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514
Here, 60 is the patient id and 971:1.000000 988:1.000000 1648:1.000000 1717:1.000000 2798:0.364078 3005:0.367953 3049:0.013514 is the feature for the patient with id 60.

Save the file as "test_features.txt" and save it inside the folder deliverables

input:
output: X_train,Y_train,X_test
'''
def my_features():
	#TODO: complete this
    #X_train,Y_train = load_svmlight_file("C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/features_svmlight.train")
    X_train,Y_train = utils.get_data_from_svmlight("C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/features_svmlight.train")
   
    events = pd.read_csv(r'C:\Users\Xiaojun\Desktop\omscs\CSE6250\hw1\data\test\events.csv')
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    
    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(r'C:\Users\Xiaojun\Desktop\omscs\CSE6250\hw1\data\test\event_feature_map.csv')
    
    patient_features = create_features(events, feature_map)
    save_svmlight(patient_features)
    
    #X_test,Y_holder = load_svmlight_file('C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.txt')
    X_test,Y_holder = utils.get_data_from_svmlight('C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.txt')
    
    return X_train,Y_train,X_test


'''
You can use any model you wish.

input: X_train, Y_train, X_test
output: Y_pred
'''
def my_classifier_predictions(X_train,Y_train,X_test):
	#TODO: complete this
    SVM = LinearSVC()
    SVM.fit(X_train, Y_train)   
	
    clf = CalibratedClassifierCV(SVM) 
    clf.fit(X_train, Y_train)
    y_proba = clf.predict_proba(X_test)
    utils.generate_submission("C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.txt",y_proba)
    return SVM.predict(X_test)


def main():
	X_train, Y_train, X_test = my_features()
	Y_pred = my_classifier_predictions(X_train,Y_train,X_test)
	#utils.generate_submission("C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/test_features.txt",Y_pred)
	#The above function will generate a csv file of (patient_id,predicted label) and will be saved as "my_predictions.csv" in the deliverables folder.

if __name__ == "__main__":
    main()

	