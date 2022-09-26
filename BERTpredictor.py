import pandas as pd
import numpy as np
import pathlib
import torch
import transformers

from typing import Union,List
from transformers import BertTokenizer, BertForPreTraining
from tqdm import tqdm

MODEL_PATH=pathlib.Path().absolute().joinpath('models') # Model path

class BERTpredictor():

    def __init__(self,model=None,tokenizer=None):

        self.model=model # Placeholder for model
        self.model_tokenizer=tokenizer # Placeholder for tokenizer
        self.input_tensor_dict=None # Place holder for input tensor dict

    def load_from_web(self,model_name:str='bert-base-uncased'):

        """Model name takes string of all available base BERT models
        available at Huggingface depository at https://huggingface.co/models"""

        self.model=BertForPreTraining.from_pretrained(model_name,output_hidden_states=True)
        self.model_tokenizer=BertTokenizer.from_pretrained(model_name)

    def load_from_local(self,fp:Union[pathlib.PurePath,str,None]=None):

        """Load model from local machine."""
        if fp==None:
            self.model=torch.load(MODEL_PATH.joinpath('best_model'))
            self.model_tokenizer=torch.load(MODEL_PATH.joinpath('best_model_t'))
        else:
            if isinstance(fp,str):
                fp=pathlib.Path(fp)
            
            self.model=torch.load(MODEL_PATH.joinpath(fp))
            self.model_tokenizer=torch.load(MODEL_PATH.joinpath(fp))

    def prepare_data_for_BERT_inference(self,text:list,prep_type:str='flatten'):
    
        """Function to prepare/tokenize data for BERT inference."""
        
        #1. Flatten text list (text split into sentences) or concatenate text (join into one text)
        if prep_type=='flatten':
            text_flat=list(np.unique(np.concatenate(text).flat))
        elif prep_type=='concat':
            text_flat=' '.join(text) 
        else:
            print("Please specify prep_type on of {'flatten','concat'}")
        
        #2. TOkenize using bert tokenizer
        bert_inputs = self.model_tokenizer(text_flat,return_tensors='pt',truncation=True, padding='max_length')
        
        return bert_inputs

    def BERT_pred_one(self,token_ids:torch.Tensor,segment_ids:torch.Tensor,attention_mask_ids:torch.Tensor):
        
        #0. Initialize model
        model=self.model

        #1. Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()
        
        #2. Add dummy batch dimension
        token_ids=token_ids[None,:]
        segment_ids=segment_ids[None,:]
        
        #3. Produce output
        with torch.no_grad():
            
            outputs=model(token_ids, segment_ids)
        
            #5. Produce hidden stantes (3rd output)
            hidden_states = outputs[2]
                
            #5. Stack embedding output
            token_embeddings = torch.stack(hidden_states[-4:], dim=0) # Stack last 4 years as suggested in research (https://jalammar.github.io/illustrated-bert/)
            #print('Stacked embedding size {}.'.format(token_embeddings.size()))
                
            #6. Remove dimension 1 (batches)
            token_embeddings = torch.squeeze(token_embeddings, dim=1).sum(dim=0)
            #print('Reduced emebedding size {}.'.format(token_embeddings.size()))
            
            #7.Attention mask padded tokens
            mask=attention_mask_ids.unsqueeze(-1).expand(token_embeddings.size()).float()
            masked_embeddings = token_embeddings * mask
            #print('Mask shape {}.'.format(mask.shape))
            
            #8. Average token embeddings to get sentence/paragraph embedding
            masked_embeddings_summed = torch.mean(masked_embeddings, 0)
            #print('Swapped dimension size {}.'.format(masked_embeddings_summed.size()))
        
        return masked_embeddings_summed

    def BERT_pred_all(self,inputs:transformers.tokenization_utils_base.BatchEncoding):

        #0. Initialize model
        model=self.model
        inputs['embeddings']=torch.empty(inputs['input_ids'].shape[0],768)

        #1. Loop over input tensor
        for i in tqdm(range(len(inputs['input_ids']))):
            token_ids=inputs['input_ids'][i]
            segment_ids=inputs['token_type_ids'][i]
            attention_mask_ids=inputs['attention_mask'][i]

            #1.1 Calculate embedding
            emb=self.BERT_pred_one(token_ids,segment_ids,attention_mask_ids)

            #1.2 Update
            inputs['embeddings'][i]=emb

        #1.3 Populate placeholder
        self.input_tensor_dict=inputs

        return inputs




