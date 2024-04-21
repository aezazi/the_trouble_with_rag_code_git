#%%
#import libraries
import json
from IPython.display import Markdown, display
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv

#%%
# load environment variables

load_dotenv(find_dotenv())
pine_key = os.environ.get('PINECONE_API_KEY') 
op_key = os.environ.get('OPENAI_API_KEY')
print(pine_key)
print(op_key)

# %%
# extract text chunks from the 'positive' and 'negative' category in the en_int.json dataset

total_vector_count = 0
query_id_list = []
text_chunk_list = []
vector_id_list = []
metadata_list = []
vector_count = 0

with open('./data/en_int.json', 'r') as f:
    for i, line in enumerate(f):
        
        # each line in the file is a query along with it's data elements
        data_line = json.loads(line)
        # print(type(data_line['positive'][0]))
        # print(data_line.keys())
        # break
        
        # data_line['positive'] is a nested_list
        for nested_list in data_line['positive']:
            # each element in the nested_list is a list of text chunks 
            # each text chunk will be vectorized and will be the text we will retrieve
            # print(len(nested_list))
            for text_chunk in nested_list:
                metadata_dict = {}
                vector_id_list.append(str(vector_count))
                text_chunk_list.append(text_chunk)

                metadata_dict['query_id'] = data_line['id']
                metadata_dict['query'] = data_line['query']
                metadata_dict['text'] = text_chunk
                metadata_dict['category'] = 'positive'
                metadata_list.append(metadata_dict)
                
                vector_count += 1
        
        # unlike for the positive cateroy, the text in the negative category is not 
        # in a nested list.
        for text_chunk in data_line['negative']:
            # each element in the list is a text chunk
            # each text chunk will be vectorized and will be the text we will retrieve
            metadata_dict = {}
            vector_id_list.append(str(vector_count))
            text_chunk_list.append(text_chunk)

            metadata_dict['query_id'] = data_line['id']
            metadata_dict['query'] = data_line['query']
            metadata_dict['text'] = text_chunk
            metadata_dict['category'] = 'negative'
            metadata_list.append(metadata_dict)
            
            vector_count += 1  

if f.closed: print('closed')

#%%
#inspect data from above for consistency
id = 200
print(vector_count)
print(len(metadata_list))
print(len(text_chunk_list))
print(len(vector_id_list))
print(vector_id_list[id])
print(metadata_list[id]['text'])
display(Markdown(f"""<b>'{text_chunk_list[id]}'</b>"""))
len(text_chunk_list[id].split())

# %%
# function to embed text chunks with openAI model
def get_embedding(text_list, model='text-embedding-3-small'):
    client= OpenAI(api_key=op_key)
    embeddings = client.embeddings.create(input=text_list ,model=model)
    return embeddings.data

# %%
#embed text chunks with small and large openAI models
#openAI has a rate limit of 5000 rpm so I have to the batch the data. I found batch
#size 2000 was about the maximum I could go

batch_size = 2000
embedded_text_chunks_small = []
for i in range(0,len(text_chunk_list), batch_size):
    batch_start = i
    batch_end = min(i+batch_size, len(text_chunk_list))
    batch = text_chunk_list[batch_start : batch_end]
    print(len(batch))

    batch_embeddings_small = get_embedding(batch, 
                                           model='text-embedding-3-small')

    embedded_text_chunks_small.extend(batch_embeddings_small)

embedded_text_chunks_large = []
for i in range(0,len(text_chunk_list), batch_size):
    batch_start = i
    batch_end = min(i+batch_size, len(text_chunk_list))
    batch = text_chunk_list[batch_start : batch_end]
    print(len(batch))

    batch_embeddings_large = get_embedding(batch, 
                                           model='text-embedding-3-large')

    embedded_text_chunks_large.extend(batch_embeddings_large)



# %%
#inspect returned embeddings object
print(type(embedded_text_chunks_small))
print(len(embedded_text_chunks_small))
print(type(embedded_text_chunks_small[0]))
print(len(embedded_text_chunks_small[0].embedding))
# embedded_text_chunks_small[1000].embedding

print(type(embedded_text_chunks_large))
print(len(embedded_text_chunks_large))
print(type(embedded_text_chunks_large[0]))
print(len(embedded_text_chunks_large[0].embedding))
# embedded_text_chunks_large[1000].embedding

# %%
# extract and store just the vectors from the openAI embeddings object into a list
embedded_text_chunks_small_vectors = embedded_text_chunks_samll_vectors = [embedded_text_chunks_small[i].embedding for i in range(len(embedded_text_chunks_large))]
embedded_text_chunks_large_vectors = [embedded_text_chunks_large[i].embedding for i in range(len(embedded_text_chunks_large))]

# %%
# inspect embedding vectors
print(len(embedded_text_chunks_small_vectors))
print(len(embedded_text_chunks_small_vectors[0]))

print(len(embedded_text_chunks_large_vectors))
print(len(embedded_text_chunks_large_vectors[0]))

# %%
# create a pandas dataframe with the embedded vectors and metadata in format for Pinecone
import pandas as pd

df_small = pd.DataFrame(
    data={
        "id": vector_id_list,
        "vectors": embedded_text_chunks_small_vectors,
        "metadata": metadata_list
    })

df_large = pd.DataFrame(
    data={
        "id": vector_id_list,
        "vectors": embedded_text_chunks_large_vectors,
        "metadata": metadata_list
    })



# %%
#inspect and pickle the dataframe

display(df_large)
display(df_small)

df_small.to_pickle('./df_en_int_pos_neg_vecs_small.pkl')
df_large.to_pickle('./df_en_int_pos_neg_vecs_large.pkl')

#%%
# instantiate Pinecone client and set to using serverless option
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
pine = Pinecone(api_key=pine_key)

# %%
# check to see if indexes by these names already exist. If so delete
index_1 = 'en-int-pos-neg-small'
index_2 = 'en-int-pos-neg-large'

if index_1 in pine.list_indexes().names():
    pine.delete_index(index_1)

if index_2 in pine.list_indexes().names():
    pine.delete_index(index_2)

pine.list_indexes().names()

# %%
# create empty Pinecone indexes
import time

pine.create_index(
    name=index_1, 
    dimension=1536, 
    metric="cosine",
    spec= ServerlessSpec(cloud='aws', region='us-west-2')
)

pine.create_index(
    name=index_2, 
    dimension=3072, 
    metric="cosine",
    spec= ServerlessSpec(cloud='aws', region='us-west-2')
)

# wait for index to be ready before connecting
while not pine.describe_index(index_1).status['ready']:
    time.sleep(1)

# %%
#check that index has been created
pine.list_indexes().names()

# %%
# insert vectors inpt Pinecone indexes
batch_size = 100
row_count_small = df_small.shape[0]
for i in range(0, row_count_small, batch_size):
    start_batch = i
    end_batch = min(start_batch + batch_size, row_count_small)
    df_batch = df_small.iloc[start_batch:end_batch, :]
    # display(df_batch)

    pine.Index('en-int-pos-neg-small').upsert(vectors=zip(df_batch.id, 
                                        df_batch.vectors, df_batch.metadata))  
    

row_count_large = df_large.shape[0]
for i in range(0, row_count_large, batch_size):
    start_batch = i
    end_batch = min(start_batch + batch_size, row_count_large)
    df_batch = df_large.iloc[start_batch:end_batch, :]
    # display(df_batch)

    pine.Index('en-int-pos-neg-large').upsert(vectors=zip(df_batch.id, 
                                        df_batch.vectors, df_batch.metadata)) 

#%%
#get stats on the indexes

display(pine.Index('en-int-pos-neg-small').describe_index_stats())
display(pine.Index('en-int-pos-neg-large').describe_index_stats())



# %%

