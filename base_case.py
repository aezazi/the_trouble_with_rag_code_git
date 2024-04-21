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
# extract text chunks from the 'positive' category in the en_int.json dataset
df_base_queries_answers = pd.DataFrame(columns=['query', 'correct_answer', 
                        'correct_answer1', 'correct_answer2', 'context'])

with open('./data/en_int.json', 'r') as f:
    for i, line in enumerate(f):
        
        # each line in the file is a query along with it's data elements
        data_line = json.loads(line)
        #pandas expects each row as a list in a list, hence the empty nested temp_list
        list_temp = [[]] 
        list_temp[0].append(data_line['query'])
        list_temp[0].append(data_line['answer'])
        #note that in the dataset the 'asnwer1' key is a misspelling which I correct
        list_temp[0].append(data_line['asnwer1'])
        list_temp[0].append(data_line['answer2'])

        #flatten context text that's in a nested list
        context = [text for list in data_line['positive'] for text in list]
        list_temp[0].append(context)

        #temp one row dataframe since pandas can concatenate only pandas dfs or series to each other
        df_temp = pd.DataFrame(list_temp, columns=['query', 'correct_answer', 
                        'correct_answer1', 'correct_answer2', 'context'])
        

        df_base_queries_answers = pd.concat([df_base_queries_answers, df_temp])

        # if i == 3 : break

if f.closed: print('closed')
df_base_queries_answers.reset_index(inplace=True, drop=True)

#%%
#inspect query list
print(len(df_base_queries_answers))
df_base_queries_answers

# %%
# instantiate openAI client and set model
client= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'gpt-4-0125-preview'

#%%
llm_answers = []
for i in range(len(df_base_queries_answers)):
    question = df_base_queries_answers.loc[i, ['query']]
    context =df_base_queries_answers.loc[i, ['context']]

    # print(question)
    # print(context)
    # print('-'*40)
    
    query = f"""Use the following information : {context} to answer this question: {question}. """
    
 
    # break
    # print(context_small_vecs)
    
    response = client.chat.completions.create(
    messages=[
        {'role': 'system', 'content': 'You are an expert research assistant that answers questions'},
        {'role': 'user', 'content': query},
    ],
    model=model,
    temperature=0,
    )

    llm_answers.append(response.choices[0].message.content)
    # if i == 3: break

# %%
llm_answers
# %%

# %%
