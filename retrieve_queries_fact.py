#%%
#import libraries
import json
from IPython.display import Markdown, display
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd

#%%
# load environment variables

load_dotenv(find_dotenv())
pine_key = os.environ.get('PINECONE_API_KEY') 
op_key = os.environ.get('OPENAI_API_KEY')
print(pine_key)
print(op_key)


# %%
# function to embed text with openAI model

def get_embedding(text_list, model='text-embedding-3-small'):
    client= OpenAI(api_key=op_key)
    embeddings = client.embeddings.create(input=text_list ,model=model)
    return embeddings.data


# %%
# extract queries and answers from en_fact.json and place into dataframe

# create empty dataframe to store results
df_en_fact_queries_answers = pd.DataFrame(columns=['query', 'answer', 
                        'positive_wrong', 'positive','negative'])

with open('./data/en_fact.json', 'r') as f:
    for i, line in enumerate(f):
        
        # each line in the file is a query along with it's data elements
        data_line = json.loads(line)
        # print(data_line.keys())

        #pandas expects each row as a list in a list, hence the empty nested temp_list
        list_temp = [[]] 
        list_temp[0].append(data_line['query'])
        list_temp[0].append(data_line['answer'])
        list_temp[0].append(data_line['positive_wrong'])
        list_temp[0].append(data_line['positive'])
        list_temp[0].append(data_line['negative'])

        #temp one row dataframe since pandas can concatenate only pandas dfs or series to each other
        df_temp = pd.DataFrame(list_temp, columns=['query', 'answer', 
                        'positive_wrong', 'positive','negative'])
        

        df_en_fact_queries_answers = pd.concat([df_en_fact_queries_answers, df_temp])

        # if i == 3 : break

if f.closed: print('closed')
df_en_fact_queries_answers.reset_index(inplace=True, drop=True)

#%%
#inspect query list
print(len(df_en_fact_queries_answers))
df_en_fact_queries_answers

#%%
# embed queries with both small and large openAI model and store in lists

en_fact_queries_embed_small_list = get_embedding(df_en_fact_queries_answers['query'], 
                                        model='text-embedding-3-small')


en_fact_queries_embed_large_list = get_embedding(df_en_fact_queries_answers['query'], 
                                        model='text-embedding-3-large')

#%%
#inspect returned embeddings objects
print(type(en_fact_queries_embed_small_list))
print(len(en_fact_queries_embed_small_list))
print(type(en_fact_queries_embed_small_list[0]))
print(len(en_fact_queries_embed_small_list[0].embedding))
# en_fact_queries_embed_small_list[0].embedding

print(type(en_fact_queries_embed_large_list))
print(len(en_fact_queries_embed_large_list))
print(type(en_fact_queries_embed_large_list[0]))
print(len(en_fact_queries_embed_large_list[0].embedding))
# en_fact_queries_embed_large_list[99].embedding

# %%
# extract and store just the vectors from the openAI embeddings object into a list
queries_row_count = len(en_fact_queries_embed_large_list)

en_fact_queries_vectors_small_list = [en_fact_queries_embed_small_list [i].embedding for i in range(queries_row_count)]
en_fact_queries_vectors_large_list = [en_fact_queries_embed_large_list [i].embedding for i in range(queries_row_count)]

# %%
# inspect embedding vectors
print(len(en_fact_queries_vectors_small_list))
print(len(en_fact_queries_vectors_small_list[0]))

print(len(en_fact_queries_vectors_large_list))
print(len(en_fact_queries_vectors_large_list[0]))

#%%
# store query vector embeddings to pickle

df_query_vecs_small = pd.DataFrame(en_fact_queries_vectors_small_list)
df_query_vecs_small.to_pickle('./df_en_fact_query_vectors_small.pkl')

df_query_vecs_large = pd.DataFrame(en_fact_queries_vectors_large_list)
df_query_vecs_large.to_pickle('./df_en_fact_query_vectors_large.pkl')

#%%
# instantiate Pinecone client 
from pinecone import Pinecone
from pinecone import ServerlessSpec, PodSpec
pine = Pinecone(api_key=pine_key)

# %%
#check Pinecone indexes
pine.list_indexes().names()

# %%
#retrieve pinecone indexes 

index_en_fact_all_small= pine.Index('en-fact-all-small')
index_en_fact_all_large = pine.Index('en-fact-all-large')

#%%
# retreive k top documents from vector database

#create empty dataframe to store metadata for all queries from retrieved docs
df_retrieved = pd.DataFrame(columns=['requested_query_id', 'requested_query', 
                    'retrieval_rank','retrieved_query_id_small', 
                    'retrieved_query_small', 'category_small', 'retrieved_text_small',
                    'retrieved_query_id_large', 'retrieved_query_large',
                    'category_large','retrieved_text_large'])


# Loop over each small and large query vector
num_query_rows = df_en_fact_queries_answers.shape[0]
for i in range(num_query_rows):
    retrieved = []
    vector_small = en_fact_queries_vectors_small_list[i]
    vector_large = en_fact_queries_vectors_large_list[i]
    
    # set number of docs to retrieve and retrieve from both small and large vectors
    k = 3
    res_small = index_en_fact_all_small.query(vector=vector_small, 
                                    top_k=k, include_metadata=True)
    
    res_large = index_en_fact_all_large.query(vector=vector_large, 
                                    top_k=k, include_metadata=True)
    
    # display(res_large)
    # break

    for k in range(k):
        retrieved.append([])
        retrieved[k].append(i)
        retrieved[k].append(df_en_fact_queries_answers.iloc[i]['query'])
        retrieved[k].append(k+1)

        retrieved[k].append(int(res_small['matches'][k]['metadata']['query_id']))
        retrieved[k].append(res_small['matches'][k]['metadata']['query'])
        retrieved[k].append(res_small['matches'][k]['metadata']['category'])
        retrieved[k].append(res_small['matches'][k]['metadata']['text'])

        retrieved[k].append(int(res_large['matches'][k]['metadata']['query_id']))
        retrieved[k].append(res_large['matches'][k]['metadata']['query'])
        retrieved[k].append(res_large['matches'][k]['metadata']['category'])
        retrieved[k].append(res_large['matches'][k]['metadata']['text'])
    
    # Store retrieved docs for this query from both small and large vector index in df_temp
    df_temp = pd.DataFrame(retrieved, columns=['requested_query_id', 'requested_query', 
                    'retrieval_rank','retrieved_query_id_small', 
                    'retrieved_query_small', 'category_small', 'retrieved_text_small',
                    'retrieved_query_id_large', 'retrieved_query_large',
                    'category_large','retrieved_text_large'])
    
    # print(df_temp.shape)
    # display(df_temp)
    
    # concatenate retrieved docs for this query to df_retrieved
    df_retrieved = pd.concat([df_retrieved, df_temp])

    # if i == 2: break
    

# %%
# reindex dataframe
df_retrieved.reset_index(inplace=True, drop=True)
display(df_retrieved)
print(df_retrieved.shape)

#%%
#convert pandas dtype to integer for numeric columns
df_retrieved[['requested_query_id', 'retrieval_rank', 'retrieved_query_id_small',
              'retrieved_query_id_large']]= \
df_retrieved[['requested_query_id', 
                'retrieval_rank', 'retrieved_query_id_small',
              'retrieved_query_id_large']].astype(int)


# df_retrieved.dtypes
# df_retrieved.to_excel("docs_retrieved__pos_comparison.xlsx")
df_retrieved.to_excel("docs_retrieved_fact_comparison.xlsx")

# %%# %%
# instantiate openAI client and set model
client= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'gpt-4-0125-preview'

# %%
