import time
import pandas as pd
import numpy as np

# PLEASE USE THE GIVEN FUNCTION NAME, DO NOT CHANGE IT

def read_csv(filepath):
    '''
    TODO : This function needs to be completed.
    Read the events.csv and mortality_events.csv files. 
    Variables returned from this function are passed as input to the metric functions.
    '''
    events = pd.read_csv(filepath + 'events.csv')
    mortality = pd.read_csv(filepath + 'mortality_events.csv')

    return events, mortality

def event_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the event count metrics.
    Event count is defined as the number of events recorded for a given patient.
    '''
    #events.columns
    
    event_count = events.groupby(['patient_id'])['event_id'].count()
    merge1 = pd.merge(event_count,mortality,how='left',on='patient_id')
    out = merge1.fillna({'label':0}).groupby(by=['label']).agg({'event_id':['mean','max','min']})
    #remove multi index
    out = out['event_id'] 
    
    # event_count.shape, merge1.shape
    # merge1.columns
    # merge1['label'].value_counts(dropna=False)   
    # merge1.head(10)
    # pd.__version__
    # type(out)
    
    avg_dead_event_count = out.loc[1.0,'mean']
    max_dead_event_count = out.loc[1.0,'max']
    min_dead_event_count = out.loc[1.0,'min']
    avg_alive_event_count = out.loc[0.0,'mean']
    max_alive_event_count = out.loc[0.0,'max']
    min_alive_event_count = out.loc[0.0,'min']

    return min_dead_event_count, max_dead_event_count, avg_dead_event_count, min_alive_event_count, max_alive_event_count, avg_alive_event_count

def encounter_count_metrics(events, mortality):
    '''
    TODO : Implement this function to return the encounter count metrics.
    Encounter count is defined as the count of unique dates on which a given patient visited the ICU. 
    '''
    
    encounter_count = events.groupby(['patient_id'])['timestamp'].nunique()
    merge1 = pd.merge(encounter_count,mortality,how='left',on='patient_id')
    out = merge1.fillna({'label':0}).groupby(by=['label']).agg({'timestamp_x':['mean','max','min']})
    #remove multi index
    out = out['timestamp_x'] 
    
    avg_dead_encounter_count = out.loc[1.0,'mean']
    max_dead_encounter_count = out.loc[1.0,'max']
    min_dead_encounter_count = out.loc[1.0,'min']
    avg_alive_encounter_count = out.loc[0.0,'mean']
    max_alive_encounter_count = out.loc[0.0,'max']
    min_alive_encounter_count = out.loc[0.0,'min']

    return min_dead_encounter_count, max_dead_encounter_count, avg_dead_encounter_count, min_alive_encounter_count, max_alive_encounter_count, avg_alive_encounter_count

def record_length_metrics(events, mortality):
    '''
    TODO: Implement this function to return the record length metrics.
    Record length is the duration between the first event and the last event for a given patient. 
    '''
    events['timestamp'] = pd.to_datetime(events['timestamp'])
    rec_len = events.groupby(['patient_id'])['timestamp'].agg(lambda x:x.max()-x.min())
    rec_len = rec_len.dt.days
    merge1 = pd.merge(rec_len ,mortality,how='left',on='patient_id')
    out = merge1.fillna({'label':0}).groupby(by=['label']).agg({'timestamp_x':['mean','max','min']})
    #remove multi index
    out = out['timestamp_x'] 
    
    
    avg_dead_rec_len = out.loc[1.0,'mean']
    max_dead_rec_len = out.loc[1.0,'max']
    min_dead_rec_len = out.loc[1.0,'min']
    avg_alive_rec_len = out.loc[0.0,'mean']
    max_alive_rec_len = out.loc[0.0,'max']
    min_alive_rec_len = out.loc[0.0,'min']

    return min_dead_rec_len, max_dead_rec_len, avg_dead_rec_len, min_alive_rec_len, max_alive_rec_len, avg_alive_rec_len

def main():
    '''
    DO NOT MODIFY THIS FUNCTION.
    '''
    # You may change the following path variable in coding but switch it back when submission.
    train_path = '../data/train/'
    # train_path = 'C:/Users/Xiaojun/Desktop/omscs/CSE6250/hw1/data/train/'

    # DO NOT CHANGE ANYTHING BELOW THIS ----------------------------
    events, mortality = read_csv(train_path)

    #Compute the event count metrics
    start_time = time.time()
    event_count = event_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute event count metrics: " + str(end_time - start_time) + "s"))
    print(event_count)

    #Compute the encounter count metrics
    start_time = time.time()
    encounter_count = encounter_count_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute encounter count metrics: " + str(end_time - start_time) + "s"))
    print(encounter_count)

    #Compute record length metrics
    start_time = time.time()
    record_length = record_length_metrics(events, mortality)
    end_time = time.time()
    print(("Time to compute record length metrics: " + str(end_time - start_time) + "s"))
    print(record_length)
    
if __name__ == "__main__":
    main()

