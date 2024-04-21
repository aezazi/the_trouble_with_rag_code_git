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
# load no_info questions from excel file
df_no_info = pd.read_excel('en_int_queries_answers.xlsx', 
                           sheet_name='no_info_for_pandas', index_col=None, header=0) 
  
# %%
# inspect df_no_info
for q in df_no_info['questions']:
    print(q)
df_no_info.shape[0]

# %%
# embed no_info questions
no_info_embeddings_small= get_embedding(df_no_info['questions'], 
                                        model='text-embedding-3-small')

# %%
# extract just the vectors from embeddings object
no_info_vectors_small = [no_info_embeddings_small[i].embedding for i in range(len(no_info_embeddings_small))]
# %%
# inspect vectors
print(len(no_info_vectors_small))
print(len(no_info_vectors_small[0]))

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
index_en_int_positive_small= pine.Index('en-int-positive-small')
index_en_int_pos_neg_small = pine.Index('en-int-pos-neg-small')


#%%
# retreive k top documents from vector database

#create empty dataframe to store metadata for all queries from retrieved docs
df_retrieved = pd.DataFrame(columns=['requested_query_id', 'requested_query', 
                    'retrieval_rank','retrieved_query_id_small', 
                    'retrieved_query_small', 'category_small', 'retrieved_text_small'])


# Loop over each small vector
num_query_rows = df_no_info.shape[0]
for i in range(num_query_rows):
    retrieved = []
    vector_small = no_info_vectors_small[i]
    # vector_large = en_int_queries_vectors_large_list[i]
    
    # set number of docs to retrieve and retrieve from both small and large vectors
    k = 3
    # res_small = index_en_int_positive_small.query(vector=vector_small, 
    #                                 top_k=k, include_metadata=True)
    
    res_small = index_en_int_pos_neg_small.query(vector=vector_small, 
                                    top_k=k, include_metadata=True)
    

    for k in range(k):
        retrieved.append([])
        retrieved[k].append(i)
        retrieved[k].append(df_no_info.iloc[i]['questions'])
        retrieved[k].append(k+1)

        retrieved[k].append(int(res_small['matches'][k]['metadata']['query_id']))
        retrieved[k].append(res_small['matches'][k]['metadata']['query'])
        retrieved[k].append(res_small['matches'][k]['metadata']['category'])
        retrieved[k].append(res_small['matches'][k]['metadata']['text'])

    
    # Store retrieved docs for this query from both small and large vector index in df_temp
    df_temp = pd.DataFrame(retrieved, columns=['requested_query_id', 'requested_query', 
                    'retrieval_rank','retrieved_query_id_small', 
                    'retrieved_query_small', 'category_small', 'retrieved_text_small'])
    
    # print(df_temp.shape)
    # display(df_temp)
   
    
    # concatenate retrieved docs for this query to df_retrieved
    df_retrieved = pd.concat([df_retrieved, df_temp])
    # break

    # if i == 2: break
# %%
# reindex dataframe and inspect
df_retrieved.reset_index(inplace=True, drop=True)
display(df_retrieved)
print(df_retrieved.shape)

# %%
# instantiate openAI client
client= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'gpt-4-0125-preview'

#%%
# loop over all questions and call llm

llm_answers_no_info = []

for i, question in enumerate(df_no_info['questions']):

    context_small_vecs = [cs for cs in  df_retrieved[df_retrieved['requested_query_id'] == i]['retrieved_text_small']]
    query_small_vecs= f"""Use the following information : {context_small_vecs} to answer this question: {question}"""
    # print(question)
    # print(context_small_vecs)

    response_small = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You are an expert research assistant that answers questions'},
        {'role': 'user', 'content': query_small_vecs},
    ],
    model=model,
    temperature=0,
    )

    llm_answers_no_info.append(response_small.choices[0].message.content)


# %%
llm_answers_no_info

# %%
df_no_info['answers_pos_neg'] = llm_answers_no_info
# %%
df_no_info
# %%
df_no_info.to_excel('./no_info_answers.xlsx')

# %%
