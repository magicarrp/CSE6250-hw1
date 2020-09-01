import utils
import pandas as pd
import time


# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    
    '''
    TODO: This function needs to be completed.
    Read the events.csv, mortality_events.csv and event_feature_map.csv files into events, mortality and feature_map.
    
    Return events, mortality and feature_map
    '''

    #Columns in events.csv - patient_id,event_id,event_description,timestamp,value
    events = pd.read_csv(filepath + 'events.csv')
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    
    #Columns in mortality_event.csv - patient_id,timestamp,label
    mortality = pd.read_csv(filepath + 'mortality_events.csv')
    mortality['timestamp'] = pd.to_datetime(mortality['timestamp'])

    #Columns in event_feature_map.csv - idx,event_id
    feature_map = pd.read_csv(filepath + 'event_feature_map.csv')

    return events, mortality, feature_map


def calculate_index_date(events, mortality, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 a

    Suggested steps:
    1. Create list of patients alive ( mortality_events.csv only contains information about patients deceased)
    2. Split events into two groups based on whether the patient is alive or deceased
    3. Calculate index date for each patient
    
    IMPORTANT:
    Save indx_date to a csv file in the deliverables folder named as etl_index_dates.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, indx_date.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)

    Return indx_date
    '''
    event_last = events.groupby(['patient_id'])['timestamp'].max()
    merge1 = pd.merge(event_last,mortality,how='left',on='patient_id')

    def index_day(x):
        if x.label == 1:
            i_date = x.timestamp_y - pd.DateOffset(days=30)
        else:
            i_date = x.timestamp_x
        return i_date
    
    merge1['indx_date'] = merge1.apply(index_day,axis=1)
    indx_date = merge1[['patient_id','indx_date']]
    indx_date.to_csv(deliverables_path + 'etl_index_dates.csv', columns=['patient_id', 'indx_date'], index=False)
       
    return indx_date


def filter_events(events, indx_date, deliverables_path):
    
    '''
    TODO: This function needs to be completed.

    Refer to instructions in Q3 b

    Suggested steps:
    1. Join indx_date with events on patient_id
    2. Filter events occuring in the observation window(IndexDate-2000 to IndexDate)
    
    
    IMPORTANT:
    Save filtered_events to a csv file in the deliverables folder named as etl_filtered_events.csv. 
    Use the global variable deliverables_path while specifying the filepath. 
    Each row is of the form patient_id, event_id, value.
    The csv file should have a header 
    For example if you are using Pandas, you could write: 
        filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)

    Return filtered_events
    '''
    
    merge1 = pd.merge(events,indx_date,how='left',on='patient_id')
    merge1 = merge1[(merge1['timestamp']>=merge1['indx_date']- pd.DateOffset(days=2000)) & (merge1['timestamp']<=merge1['indx_date'])]

    filtered_events = merge1[['patient_id', 'event_id', 'value']]
    filtered_events.to_csv(deliverables_path + 'etl_filtered_events.csv', columns=['patient_id', 'event_id', 'value'], index=False)
    return filtered_events


def aggregate_events(filtered_events_df, mortality_df,feature_map_df, deliverables_path):
    
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
    
    aggregated_events.to_csv(deliverables_path + 'etl_aggregated_events.csv', columns=['patient_id', 'feature_id', 'feature_value'], index=False)
    return aggregated_events

def create_features(events, mortality, feature_map):
    
    deliverables_path = '../deliverables/'
    # deliverables_path = 'C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/'
    

    #Calculate index date
    indx_date = calculate_index_date(events, mortality, deliverables_path)

    #Filter events in the observation window
    filtered_events = filter_events(events, indx_date,  deliverables_path)
    
    #Aggregate the event values for each patient 
    aggregated_events = aggregate_events(filtered_events, mortality, feature_map, deliverables_path)

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
        
    merge2 = pd.merge(filtered_events,mortality,on='patient_id',how='left')
    merge2.fillna(value = {'label':0}, inplace=True)
    mortality = dict(zip(merge2['patient_id'],merge2['label']))

    return patient_features, mortality

def save_svmlight(patient_features, mortality, op_file, op_deliverable):
    
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
    deliverable1 = open(op_file, 'wb')
    deliverable2 = open(op_deliverable, 'wb')
    
    # test = sorted(patient_features[41], key = lambda x: x[0])
    
    out1 = ""
    for i in patient_features:
        out1 = out1 + str(mortality[i]) + ' '
        for j in sorted(patient_features[i], key = lambda x: x[0]):
            out1 = out1 + str(int(j[0])) + ':'+ str(format(j[1], '.6f')) + ' '
        out1 = out1 + '\n'
        
    out2 = ""
    for i in patient_features:
        out2 = out2 + str(int(i)) + ' ' + str(mortality[i]) + ' '
        for j in sorted(patient_features[i], key = lambda x: x[0]):
            out2 = out2 + str(int(j[0])) + ':'+ str(format(j[1], '.6f')) + ' '
        out2 = out2 + '\n'
        
    #print(out2)
    deliverable1.write(bytes(out1,'UTF-8')); #Use 'UTF-8'
    deliverable2.write(bytes(out2,'UTF-8'));

def main():
    train_path = '../data/train/'
    # train_path = 'C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/data/train/'
    events, mortality, feature_map = read_csv(train_path)
    patient_features, mortality = create_features(events, mortality, feature_map)
    save_svmlight(patient_features, mortality, '../deliverables/features_svmlight.train', '../deliverables/features.train')
    # save_svmlight(patient_features, mortality, 'C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/features_svmlight.train', 'C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/deliverables/features.train')
    

if __name__ == "__main__":
    main()