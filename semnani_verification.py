#%%
#import libraries
import json
from IPython.display import Markdown, display
from openai import OpenAI
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datetime import date


#%%
# load environment variables

load_dotenv(find_dotenv())
pine_key = os.environ.get('PINECONE_API_KEY') 
op_key = os.environ.get('OPENAI_API_KEY')
print(pine_key)
print(op_key)
today = date.today()
print(today)

# %%
# read en_fact.json into a list of dictionaries

with open('./data/en_fact.json', 'r') as f:
    en_fact_data = [json.loads(line) for line in f]

if f.closed: print('closed')

# %%
# inspect en_fact_data
print(len(en_fact_data))
for t in en_fact_data[99]['positive_wrong']:
    print(t)


# %%
# instantiate openAI client and set model
client= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
model = 'gpt-4-0125-preview'


# %%
# function to generate answers from positive_wrong retrieved data and place in dataframe
def create_adverserial_QA(adverserial_data = en_fact_data, model = 'gpt-4-0125-preview'):
    
    df_qa_pos_wrong = pd.DataFrame(columns=['question', 'context_pos_wrong','answer_pos_wrong'])
    
    for data_line in adverserial_data:
        
        question = data_line['query']
        
        context = data_line['positive_wrong']
        # print(question)
        # print(context)
        query_original_question = f"""Use the following information : {context} to answer this question: {question}
                                    Use only the information provided. Do not use your prior knowledge. Do not correct any errors"""
        
        response_pos_wrong = client.chat.completions.create(
            messages=[
                {'role': 'system', 
                'content': 'You are an expert research assistant that answers questions'},
                {'role': 'user', 'content': query_original_question},
            ],
        model=model,
        temperature=0
        )
        temp_dict = {"question": question, 
                    "response": response_pos_wrong.choices[0].message.content
                    }
        
        answer_pos_wrong = response_pos_wrong.choices[0].message.content
        # print(answer_pos_wrong)
        df_temp = pd.DataFrame([[question, context, answer_pos_wrong]],
                            columns=['question', 'context_pos_wrong','answer_pos_wrong'])
        df_qa_pos_wrong = pd.concat([df_qa_pos_wrong, df_temp])
        
    return df_qa_pos_wrong
#%%
#create dataframe with positive_wrong answers
df_qa_pos_wrong = create_adverserial_QA() 
df_qa_pos_wrong.reset_index(inplace=True, drop=True)

#%%
#inspect df_qa_pos_wrong
display(df_qa_pos_wrong['answer_pos_wrong'])
print(df_qa_pos_wrong.shape)

# %%
# few shot examples for extracting specific claims from wrong answers
few_shot_examples_extract_claims = [
            f"""
            question : When was chinese new year this year ?
            response: It was on February 5th.
            Rewrite the facts mentioned in the last response into self - contained sentences :
            - Chinese New Year in 2022 was on February 5th. The year of the results is "2022".
            """,
            f"""
            question : Do you know about Queen Elizabeth II? I was just reading about her .
            response : Queen Elizabeth II is the current monarch of the United Kingdom and the
            Commonwealth realms . She was born in 1926 and became queen in 1952.
            Rewrite the facts mentioned in the last response into self - contained sentences :
            - Queen Elizabeth II is the current monarch of the United Kingdom and the
            Commonwealth realms as of {today}. The year of the results is " recent ".
            - Queen Elizabeth II was born in 1926. The year of the results is "1926".
            - Queen Elizabeth II became queen in 1952. The year of the results is "1952".
            """,
            f"""
            question : How are you doing today ?
            response : As an artificial intelligence , I don 't have feelings or personal experiences ,
            so I don 't have good days or bad days . However , I'm fully operational and ready
            to assist you with any information or fact checking tasks you need . What can I help you with
            today ?
            Rewrite the facts mentioned in the last response into self - contained sentences :
            Nothing .
            """,
            f"""
            question: who is the current Prime Minister of the United Kingdom?
            response: As of my cut-off knowledge date of 2021, the Prime Minister of the United Kingdom was Boris Johnson. 
            I apologize for not being able to provide information about any changes that may have occurred after 2021.
            Rewrite the facts mentioned in the last response into self - contained sentences :
            - Boris Johnson was the Prime Minister of the United Kingdom as of 2021. The year of the results is "2021".
            """,
            f"""
            question: When was chinese new year this year ?
            response: It was on February 5th.
            Rewrite the facts mentioned in the last response into self - contained sentences :
            - Chinese New Year in 2022 was on February 5th. The year of the results is "2022".
            """,
            """
            question: tell me about Michelle Yeoh
            response: Yes, I have heard about "Everything Everywhere All at Once". 
            It is a 2022 American absurdist comedy-drama film directed by the Daniels and 
            produced by Anthony and Joe Russo. The movie stars Michelle Yeoh as a Chinese-American
            immigrant who must connect with parallel universe versions of herself to prevent a 
            powerful being from destroying the multiverse. The film received critical acclaim for 
            its imagination, visual effects, humor, direction, editing, acting, and handling of 
            themes such as existentialism, nihilism, and Asian-American identity
            Rewrite the facts mentioned in the last response into self-contained sentences:
            - "Everything Everywhere All at Once" is a 2022 American absurdist comedy-drama film. The year of the results is "2022".
            - The movie "Everything Everywhere All at Once" is directed by the Daniels. The year of the results is "2022".
            - The movie "Everything Everywhere All at Once" is produced by Anthony and Joe Russo. The year of the results is "2022".
            - Michelle Yeoh plays the role of a Chinese-American immigrant in the movie "Everything Everywhere All at Once". The year of the results is "2022".
            - The movie "Everything Everywhere All at Once" received critical acclaim. The year of the results is "2022".
            f"""
            
            ]



#%%
# function to extract specifc claims from wrong answers

def extract_claims(dataframe, few_shot_examples, model = 'gpt-4-0125-preview'):
    extracted_claims = []

    for i in range(len(dataframe)):
        question = dataframe.loc[i, 'question']
        llm_answer_pos_wrong = dataframe.loc[i, 'answer_pos_wrong']

        query_extract_claims = f"""The response to this question: {question} was {llm_answer_pos_wrong}.
        rewrite the response {llm_answer_pos_wrong} into self - contained sentences . Exclude opinions,
        or subjective statements. Use only the information provided. Do not use your prior knowledge. 
        Do not correct any errors
        Here are some exmaple: {few_shot_examples}"""
        
        response_extract_claims = client.chat.completions.create(
            messages=[
                {'role': 'system', 
                'content': f"""You are an expert research assistant.
                            today's date is {today}"""},
                {'role': 'user', 'content': query_extract_claims},
            ],
            model=model,
            temperature=0,
            )
        claim = response_extract_claims.choices[0].message.content
        extracted_claims.append(claim)
        
    return extracted_claims
# %%
# extract claims from wrong answers
claims = extract_claims(df_qa_pos_wrong, few_shot_examples_extract_claims)
#%%
for c in claims:
    display(Markdown(f"""<b>'{c}'</b>"""))

#%%
#insert specific claims extracted from wrong answers into dataframe
# df_qa_pos_wrong.insert(3, "pos_wrong_claims",claims,  allow_duplicates=False)
display(df_qa_pos_wrong)
df_qa_pos_wrong.to_pickle('./df_qa_pos_wrong.pkl')


#%%
#few shot examples for verifying accuracy of claims

few_shot_examples_verify_claims = [
        f"""
        question: "when was the last eruption of Mauna Loa?"
        answer: "The last eruption of Mauna Loa started on March 25, 1984."
        you search the internet to fact check the claim: "The last eruption of Mauna Loa started on March 25, 1984."
    

        [You find these articles:
            Title: 2022 eruption of Mauna Loa
            Article: When active, Mauna Loa tends to produce "voluminous, fast-moving lava flows" 
            of the Hawaiian or effusive eruption type rather than more explosive phreatic or 
            Plinian eruptions, though it has produced explosive eruptions between 300 and 1,000 years
              ago. Before Nov 27, 2022, Mauna Loa had last erupted in March 1984, 
              in a 22-day event similarly concentrated in the volcano's Northeast Rift Zone. 
              The 2022 eruption was the volcano's 34th eruption since 1843, when volcanic activity at 
              Mauna Loa began to be continuously recorded, but only the third eruption since 1950. 
              The 38-year span between the 1984 and 2022 eruptions was Mauna Loa's longest period 
              of quiescence on record.

            Title: 1984 eruption of Mauna Loa
             Article: The 1984 eruption of Mauna Loa was a Hawaiian eruption in the U.S. state 
             of Hawaii that lasted from March 25 to April 15, 1984. 
             It ended a 9-year period of quiescence at the volcano and continued for 22 days, 
             during which time lava flows and lava fountains issued from the summit caldera and 
             fissures along the northeast and southwest rift zones. Although the lava threatened Hilo, 
             the flow stopped before reaching the outskirts of town.

        ]
        Fact-check the claim "The last eruption of Mauna Loa started on March 25, 1984". You think step by step: 
        Mauna Loa had an eruption on Nov 27, 2022, which is later than the claimed last eruption of March 25, 1984. 
        So the last eruption of Mauna Loa was not on March 25, 1984. So the fact-checking result is "REFUTES".

        You respond in json with the following format: 
        {{"original_question": "when was the last eruption of Mauna Loa?",
        "claimed_answer": ""The last eruption of Mauna Loa started on March 25, 1984."
        "result:" "REFUTES"
        "verify_answer_llm": the last eruption of Mauna Loa was not on March 25, 1984. The last eruption of Mauna Loa started on March 25, 1984
        "sources": each of the articles you found in json with the following format:
            {{"title": Title of the article}}, 
            {{"url": The url of the article you found}},
            {{"text": the text of the article you found}}
        }}
        """,

        f"""
        question: "What material was used to build the Eiffel Tower?"
        answer: "The Eiffel Tower was built with iron."
        you search the internet to fact check the claim: "The Eiffel Tower was built with iron."

        [you find these articles:
            Title: Vladimir Tatlin
            Article: The monument was to be a tall tower made of iron, glass and steel which 
            would have dwarfed the Eiffel Tower in Paris (the Monument to the Third International was a third taller at 400 meters high). 
            Inside the iron-and-steel structure of twin spirals, the design envisaged three building blocks, 
            covered with glass windows, which would rotate at different speeds (the first one, a cube, 
            once a year; the second one, a pyramid, once a month; the third one, a cylinder, once a day). 
            The entire building was to house the executive and legislature of the Comintern, and be a 
            central area for the creation and dissemination of propaganda. For financial and practical 
            reasons, however, the tower was never built.

            Title: Eiffel Tower
            Article: Tower on the Champ de Mars in Paris, France The Eiffel Tower is a wrought-iron 
            lattice tower on the Champ de Mars in Paris, France. It is named after the engineer 
            Gustave Eiffel, whose company designed and built the tower. 
            Locally nicknamed "La dame de fer" (French for "Iron Lady"), it was constructed from
            1887 to 1889 as the centerpiece of the 1889 World's Fair. Although initially 
            criticised by some of France's leading artists and intellectuals for its design, 
            it has since become a global cultural icon of France and one of the most recognisable 
            structures in the world.
        ]
        Fact-check the claim "The Eiffel Tower is made of iron". You think step by step:
        The Eiffel Tower is a wrought-iron lattice tower, so it is made of iron. 
        So the fact-checking result is "SUPPORTS".

        You respond in json with the following format: 
        {{"original_question": "What material was used to build the Eiffel Tower?"
        "claimed_answer": "The Eiffel Tower was built with iron."
        "result": "SUPPORTS"
        "verify_answer_llm": "The Eiffel Tower was built with iron."
        "sources": each of the articles you found in json with the following format:
            {{"title": Title of the article}}, 
            {{"url": The url of the article you found}},
            {{"text": the text of the article you found}}
        }}
        

        """,

       f"""
        question: "on what date was the Chinese New Year in 2020?"
        answer: "The Chinese New Year in 2020 was on February 10th"
        you search the internet to fact check the claim: "The Chinese New Year in 2020 was on February 10th"
        [you find these articles:
            Title: Chinese government response to COVID-19
            Article: 2020 Chinese New Year. The Wuhan government, which announced a number of new measures such as cancelling the Chinese New Year celebrations, in addition to measures such as checking the temperature of passengers at transport terminals first introduced on 14 January.  The leading group decided to extend the Spring Festival holiday to contain the outbreak.


            Title: Chinese calendar
            Article: Chinese New Year. The date of the Chinese New Year accords with the patterns of the lunisolar calendar and hence is variable from year to year. However, two general rules govern the date. Firstly, Chinese New Year transpires on the second new moon following the December solstice. If there is a leap month after the eleventh or twelfth month, then Chinese New Year falls on the third new moon after the December solstice.

        ]
        
        Fact-check the claim "The Chinese New Year in 2020 was on February 10th". 
        You think step by step:
        There is no information about the dates of Chinese New Year in 2020 in these articles. 
        So the fact-checking result is "NOT ENOUGH INFO".


        You respond in json with the following format: 
        {{"original_question": "on what date was the Chinese New Year in 2020?"
        "claimed_answer": "The Chinese New Year in 2020 was on February 10th"
        "result": "NOT ENOUGH INFO"
        "verify_answer_llm": "I'm not sure when chinese new year was in 2020. But I am attaching some information that might be helpful"
        "sources": each of the articles you found in json with the following format:
            {{"title": Title of the article}}, 
            {{"url": The url of the article you found}},
            {{"text": the text of the article you found}}
        }}
        """,

]

#%%
# function to verify claims

def verify_claims(dataframe, few_shot_examples, model = 'gpt-4-0125-preview'):
    verified = []

    for i in range(len(dataframe)):
        question = dataframe.loc[i, 'question']
        pos_wrong_claims = dataframe.loc[i, 'pos_wrong_claims']

        query_verify_claims = f"""
        The answer to this question: {question} was: {pos_wrong_claims}. For each claim in: {pos_wrong_claims}
        you search the internet to obtain articles that would support or refute that claim , and output one of 
        "SUPPORTS ", " REFUTES ", or " NOT ENOUGH INFO ". Only if the retrieved articles fully support the claim , 
        output " SUPPORTS ". make sure you remember the url of the articles you found and  the text
        you used to refute or support the claim. please respond in json.

        Here are some examples: {few_shot_examples}
        """

        response_verify_claims = client.chat.completions.create(
            messages=[
                {'role': 'system', 
                'content': f"""You are an expert research assistant and fact checker that answers questions.
                            You will be given a question and an answer to the question. Each answer may contain
                            one or more claims. You will fact check the claims. today's date is {today}."""},
                {'role': 'user', 'content': f"""{query_verify_claims}"""},
                
            ],
            model=model,
            temperature=0,
            response_format={"type": "json_object"}
            )
        verify_answer_llm = response_verify_claims.choices[0].message.content
        print(i)
        print(verify_answer_llm)
        verified.append(verify_answer_llm)

        # break
    return verified



#%%
# verify claims
verified = verify_claims(df_qa_pos_wrong, few_shot_examples_verify_claims)


#%%
#inspect verified json
print(len(verified))
type(verified)
display(verified[23])
# %%
# convert json to Pyhton dictionary and place in list
verified_dict = [json.loads(j) for j in verified]

# %%
# verify that llm returned the correct json format
len(verified_dict)
print(verified_dict[0].keys())
for i, dict in enumerate(verified_dict):
    keys = dict.keys()
    if 'original_question' not in keys: print(f"""row {i} 'original_question' key anomaly""")
    if 'claimed_answer' not in keys: print(f"""row {i} 'claimed_answer' key anomaly""")
    if 'result' not in keys: print(f"""row {i} 'result' key anomaly""")
    if 'verify_answer_llm' not in keys: print(f"""row {i} 'verify_answer_llm' key anomaly""")
    if 'sources' not in keys: print(f"""row {i} 'sources' key anomaly""")
    

# %%
df_verified_pos_wrong = pd.DataFrame(columns=['question', 'llm_verified_dict',
                                              'pos_wrong_claims', 'llm_verification_result',
                                               'llm_verification_sources', 'llm_verification_text'])

for d in verified_dict:

    # if 'claimed_answer' not in d.keys(): continue

    question = d['original_question']
    llm_verified_dict = d

    try:
        pos_wrong_claims = d['claimed_answer']
    except KeyError:
        try: 
            pos_wrong_claims = d['claimed_answers']
            print(type(d['claimed_answers']))
            print(d['claimed_answers'])
            continue
        except Exception as e:
            print(e) 
    else:
        pos_wrong_claims = d['claimed_answer']
    llm_verification_result = d['result']
    llm_verification_sources = d['sources']

    sources_text = []
    for source in d['sources']:
        # print(source['text'])
        sources_text.append(source['text'])
    
    llm_verification_text = sources_text
    df_temp = pd.DataFrame([[question, llm_verified_dict, pos_wrong_claims, llm_verification_result,
                            llm_verification_sources, llm_verification_text]], 
                            columns=['question', 'llm_verified_dict',
                                              'pos_wrong_claims', 'llm_verification_result',
                                               'llm_verification_sources', 'llm_verification_text'])
    # display(df_temp)
    df_verified_pos_wrong= pd.concat([df_verified_pos_wrong, df_temp])
    df_verified_pos_wrong.reset_index(inplace=True, drop=True)
    # print(df_verified_pos_wrong.shape)
    # break

#%%
# inspect and df save to pickle


display(df_verified_pos_wrong)

# df_verified_pos_wrong.to_pickle('./df_verified_pos_wrong.pkl')

# %%
# update answers with text retrieved in verification stage
def llm_update_answers(dataframe, model = 'gpt-4-0125-preview'):
    updated_answers = []

    for i  in range(len(dataframe)):
        question = dataframe.loc[i, 'question']
        context = dataframe.loc[i, 'llm_verification_text']

        query_original_question = f"""Use the following information : {context} to answer this question: {question}
                                    Use only the information provided. Do not use your prior knowledge. Do not correct any errors"""
        
        response_updated_answer = client.chat.completions.create(
            messages=[
                {'role': 'system', 
                'content': 'You are an expert research assistant that answers questions'},
                {'role': 'user', 'content': query_original_question},
            ],
        model=model,
        temperature=0
        )

        updated_answer = response_updated_answer.choices[0].message.content
        display(updated_answer)
        updated_answers.append(updated_answer)
        # break
    return updated_answers

# %%
#get updated answers
updated_answers = llm_update_answers(df_verified_pos_wrong)

#%%
#inspect udated answers
display(updated_answers)
len(updated_answers)

#%% 
#get ground truth answers from dataset

ground_truth_answers = []
with open('./data/en_fact.json', 'r') as f:
   for i, line in enumerate(f):
        
        # each line in the file is a query along with it's data elements
        data_line = json.loads(line)
        if i == 59: continue
        ground_truth_answers.append(data_line['answer'])

if f.closed: print('closed')
print(len(ground_truth_answers))


# %%
# create final dataframe with updated llm answers and ground truth answers
df_verified_pos_wrong_updated_answers = df_verified_pos_wrong.assign(updated_llm_answers = updated_answers,
                                                                     ground_truth_answers = ground_truth_answers)

#%%
display(df_verified_pos_wrong_updated_answers)
df_verified_pos_wrong_updated_answers.to_pickle('./df_verified_pos_wrong_updated_answers.pkl')
df_verified_pos_wrong_updated_answers.to_excel('./verified_pos_wrong_updated_answers.xlsx')

# %%
ground_truth_answers = []
with open('./data/en_fact.json', 'r') as f:
   for i, line in enumerate(f):
        
        # each line in the file is a query along with it's data elements
        data_line = json.loads(line)
        if i == 59: continue
        ground_truth_answers.append(data_line['answer'])

if f.closed: print('closed')
# %%
