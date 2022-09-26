from typing import Union,List
from  DataPreper import *
from BERTpredictor import *

import pandas as pd
import numpy as np
import pathlib
import torch
import transformers
import pickle

QUESTION_RAW_PATH=pathlib.Path().absolute().joinpath('questions') # Path to questions folder
QUESTION_EMBEDDINGS=pathlib.Path().absolute().joinpath('questions_embeddings') # Path to questions folder
QUESTIONS_INDEXED=pathlib.Path().absolute().joinpath('questions_indexed') # Path to questions folder
QUERY_PATH=pathlib.Path().absolute().joinpath('corpus') # Path to queries folder
QUERIES_IDNEXED=pathlib.Path().absolute().joinpath('corpus_indexed') # Path to queries folder


class QueryMatcher():

    def __init__(self,model,tokenizer,load_embeddings:bool=False,load_indexes:bool=True):
        
        self.queries=None # QUeries placeholder
        self.queries_index=None # Queries index
        
        self.questions=None #Placeholder raw questions
        self.question_embeddings=None #Placeholder for question embeddings
        self.question_index=None #Placeholder for question file index

        # Whether to load embeddings
        if load_embeddings:
            self.question_embeddings=self.load_files(QUESTION_EMBEDDINGS,'questions_embedings.pickle')

        # Whether to load indexes
        if load_indexes:
            self.question_index=self.load_files(QUESTIONS_INDEXED,'questions_indexed.pickle')
            self.queries_index=self.load_files(QUERIES_IDNEXED,'corpus_indexed.pickle')

        self.model=model # Embedding model
        self.tokenizer=tokenizer #Load 

    def parse_questions_files(self,question_path:Union[pathlib.PurePath,str]=QUESTION_RAW_PATH,to_sentence:bool=True,save_index:bool=True):
        
        """Extract questions from text files in provided folder path."""

        #1. Parse questions
        parser=DataPrepper()
        questions_list=parser.txt_to_lists(question_path,to_sentence=to_sentence)

        #2. Populate placeholders
        self.questions=questions_list
        self.question_index=parser.data_index

        parser.clear_vars()

        #3. Save indexed question pickle
        if save_index:
            QUESTIONS_INDEXED.mkdir(parents=False, exist_ok=True)
            self.save_files(QUESTIONS_INDEXED.joinpath('questions_indexed.pickle'),self.questions_index)

        return questions_list

    def parse_queries_files(self,query_path:Union[pathlib.PurePath,str]=QUERY_PATH,to_sentence:bool=True,save_index:bool=True):
        
        """Extract quries from text files in provided folder path"""

        #1.Parse queries
        parser=DataPrepper()
        queries_list=parser.txt_to_lists(query_path,to_sentence=True)
        
        #2. Populate placeholders
        self.queries=queries_list
        self.queries_index=parser.data_index

        parser.clear_vars()

        #3. Save indexed query pickle
        if save_index:
            QUERIES_IDNEXED.mkdir(parents=False, exist_ok=True)
            self.save_files(QUERIES_IDNEXED.joinpath('corpus_indexed.pickle'),self.queries_index)

        return queries_list

    def calc_question_embeddings(self,question_list:List[list],save:bool=False):

        """Calculate embeddings for questions list."""

        #1. Prepare queries for embedding calculation
        bert_predictor=BERTpredictor(self.model,self.tokenizer)
        questions_dict=bert_predictor.prepare_data_for_BERT_inference(question_list,prep_type='flatten')

        #2. Calculate question emebeddings
        questions_dict_emb=bert_predictor.BERT_pred_all(questions_dict)
        self.question_embeddings=questions_dict_emb

        #3. Save embeddings
        if save:
            QUESTION_EMBEDDINGS.mkdir(parents=False, exist_ok=True)
            self.save_files(QUESTION_EMBEDDINGS.joinpath('questions_embedings.pickle'),self.question_embeddings)
            
        return questions_dict_emb

    def save_files(self,fp,f):

        """Save file with filename f to filepath fp"""
        
        #1. Save embeddings
        if isinstance(fp,str):
            fp=pathlib.Path(fp)

        with open(fp, 'wb') as handle:
            pickle.dump(f, handle)


    def load_files(self,fp,f):

        """Load file with filename f from path fp"""

        #1. Load embeddings
        if isinstance(fp,str):
            fp=pathlib.Path(fp)

        with open(fp.joinpath(f), 'rb') as handle:
            b = pickle.load(handle)

        return b

    def match_query(self,query):

        """Find best matching question to text (query)."""

        #1. Convert text into list of sentences (or leave as it is)
        query=nltk.tokenize.sent_tokenize(' '.join(query))

        #2. Queries list
        if self.question_embeddings==None:
            print('Please calculate/load question embeddings.')
            return None

        #3. Prep for inference
        bert_predictor=BERTpredictor(model=self.model,tokenizer=self.tokenizer)
        query_dict=bert_predictor.prepare_data_for_BERT_inference(query,prep_type='concat')

        #4. Find query embedding
        token_ids=query_dict['input_ids'][0]
        segment_ids=query_dict['token_type_ids'][0]
        attention_mask_ids=query_dict['attention_mask'][0]

        query_embedding=bert_predictor.BERT_pred_one(token_ids,segment_ids,attention_mask_ids)

        #4. Look for match
        res_list=[]
        for i in range(self.question_embeddings['input_ids'].shape[0]):
            
            sim=self.compare_text(self.question_embeddings['embeddings'][i],query_embedding)
            res_list.append(('question_{}'.format(i),sim))

        #5. Get maximum match
        res_list.sort(key = lambda x: x[1],reverse=True)

        #6. Get actual maximum query match
        best_query=self.question_index[0]['text'][int(res_list[0][0].split('_')[-1])]

        return res_list,best_query

    def match_queries(self,queries:List[list],prep_type:str='flatten'):

        """Find best matching questions for list of queries."""

        #0. INitalized bert predictor
        bert_predictor=BERTpredictor(model=self.model,tokenizer=self.tokenizer)

        #1. Question list check
        if self.question_embeddings==None:
            print('Please calculate/load question embeddings.')
            return None

        res_dict={}
        #2. Loop over texts
        for i in range(len(queries)):
            
            res_dict[self.queries_index[i]['filename']]={}
            #2.1 Loop over lines and evaluate one after another
            for j in range(len(queries[i])):
                #2.2 Tokenize
                query_list=queries[i][:j+1]

                query_list_bert=bert_predictor.prepare_data_for_BERT_inference(query_list,prep_type=prep_type)
                query_embedding=bert_predictor.BERT_pred_one(query_list_bert['input_ids'][0],query_list_bert['token_type_ids'][0],query_list_bert['attention_mask'][0])
                
                #2.3 Calc cos similarity
                res_list=[(self.question_index[0]['filename']+'_{}'.format(i),self.compare_text(query_embedding,emb)) for i,emb in enumerate(self.question_embeddings['embeddings'])]
                res_list.sort(key = lambda x: x[1],reverse=True)

                #2.4 Update results dict
                res_dict[self.queries_index[i]['filename']]['{}_sentences'.format(j)]=res_list

        return res_dict

    def match_queries_to_df(self,match_res:dict,aggregation_type:str='average'):

        """Convert match query dict to df. Aggregation takes one of {averagre,max,last}"""

        #1. Prepare match results (from dict to dif)
        L=[]
        for k_0,v_0 in match_res.items():
            l = [(k, *t) for k, v in v_0.items() for t in v]
            df_temp = pd.DataFrame(l, columns=['n_of_sentences','question_file_id','question_similarity_score'])
            df_temp['question_id']=df_temp['question_file_id'].apply(lambda x: x.split('_')[-1])
            df_temp['log_file']=k_0
            df_temp['question_file']=df_temp['question_file_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
            df_temp['question_similarity_score']=df_temp['question_similarity_score'].astype(float)
            L.append(df_temp)
        L_df=pd.concat(L,axis=0)

        #2. Aggregate results
        if aggregation_type=='average':
            L_df_res=L_df.groupby(['question_file','question_id'])['question_similarity_score'].mean().to_frame().sort_values('question_similarity_score',ascending=False)
        
        elif aggregation_type=='last':
            L_df.sort_values('n_of_sentences',ascending=True,inplace=True)
            L_df_res=L_df.groupby(['question_file','question_id'])['question_similarity_score'].last().to_frame()

        elif aggregation_type=='max':
            L_df_res=L_df.groupby(['question_file','question_id'])['question_similarity_score'].max().to_frame()
        else:
            print('Please specify correct aggregation type i.e in {averagre,max,last}.')


        return L_df_res

    def compare_text(self,emb1,emb2):
    
        """Calculate similarity between embeddings"""
        
        cos=torch.nn.CosineSimilarity(dim=0)
        
        diff_emb = cos(emb1, emb2)
    
        return diff_emb








    

    
