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
