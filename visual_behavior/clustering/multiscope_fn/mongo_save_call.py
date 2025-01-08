

###############################################################################################
##### Save a dataframe entries to MONGO and retrieve them later
##### code from Doug: https://gist.github.com/dougollerenshaw/b2ddb9f5e26f33059001f21bae49ea2b
###############################################################################################

#%% save the dataframe to mongo (make every row a 'document' or entry)¶
# This might be slow for a big table, but you only need to do it once

df = stim_response_dfs

from visual_behavior import database


conn = database.Database('visual_behavior_data')
collection_name = f'{project_codes}_{session_numbers}' # 'test_table'
collection = conn['ophys_population_analysis'][collection_name]

print(collection_name)

for idx,row in df.iterrows():
    # run every document through the 'clean_and_timestamp' function.
    # this function ensures that all datatypes are good for mongo, then adds an additional field with the current timestamp
    document_to_save = database.clean_and_timestamp(row.to_dict()) 
    database.update_or_create(
        collection  = collection, # this is the collection you defined above. it's the mongo equivalent of a table
        document = document_to_save, # save the document in mongo. Think of every document as a row in your table
        keys_to_check = ['ophys_experiment_id', 'ophys_session_id', 'stimulus_presentations_id', 'cell_specimen_id'], # this will check to make sure that duplicate documents don't exist
    )
    
# close the connection when you're done with it
conn.close()




# Now get data from the table
# Search for only the subset of data you're interested in

# Note use the following to get all entries:
# collection.find({})
    
%%time
desired_oeid = 2
# the find function returns a cursor. 
# to actually get the data, it must be cast to a list.
# to make it conviently readable, then cast to a dataframe

conn = database.Database('visual_behavior_data')
collection_name = 'test_table'
collection = conn['ophys_population_analysis'][collection_name]

df_subset = pd.DataFrame(list(collection.find({'ophys_experiment_id':desired_oeid})))

# close the connection when you're done with it
conn.close()

df_subset