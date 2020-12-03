import tensorflow as tf
import pandas as pd


def convert_tb_events_to_dataframe(events_path):
    
    all_data = {}
    
    tb_summary_iterator = tf.compat.v1.train.summary_iterator(events_path)
    
    for e in tb_summary_iterator:
        for v in e.summary.value:
            if v.tag not in all_data.keys():
                all_data[v.tag] = []
            
            all_data[v.tag].append(tf.squeeze(tf.make_ndarray(v.tensor)).numpy())
            
    all_data_df = pd.DataFrame(all_data)
    
    return all_data_df


def convert_training_loop_to_dataframe(training_loop_path, total_rounds_count, total_clients_count):
    
    all_data = {}
    
    # Need to populate in advance all the keys (columns in DF) and values at each round
    for i in range(total_clients_count):
        key_name = 'client_' + str(i)
        
        all_data[key_name] = []
        
        for _ in range(total_rounds_count):
            all_data[key_name].append(0)
    
    with open(training_loop_path, 'r') as training_loop_file:
        
        current_line = training_loop_file.readline()
        
        
        while current_line:
            current_line_elements = current_line.split()
            
            if len(current_line_elements) > 2 and current_line_elements[2].startswith('start!'):
                current_round = current_line_elements[1]
            elif len(current_line_elements) > 5 and current_line_elements[5].startswith('finished'):
                key_name = 'client_' + str(current_line_elements[2]) # client serial
                
                all_data[key_name][int(current_round)-1] = float(current_line_elements[-1]) # client loss value
            
            current_line = training_loop_file.readline()
    
    all_data_df = pd.DataFrame(all_data)
    
    return all_data_df
