import pandas as pd
import numpy as np
import pathlib
import torch
import transformers

from typing import Union,List
from transformers import BertTokenizer, BertForPreTraining
from collections import defaultdict
from tqdm import tqdm
from transformers import AdamW


MODEL_PATH=pathlib.Path().absolute().joinpath('models') # Model path

class BERTtuner():

    def __init__(self):

        self.model=None # Placeholder for model
        self.model_tokenizer=None # Placeholder for tokenizer
        self.bert_inputs=None # Placeholder for BERT input tensor dict
    
    def load_from_web(self,model_name:str='bert-base-uncased'):

        """Model name takes string of all available base BERT models
        available at Huggingface depository at https://huggingface.co/models"""

        self.model=BertForPreTraining.from_pretrained(model_name,output_hidden_states=True)
        self.model_tokenizer=BertTokenizer.from_pretrained(model_name)

        print('Model {} had been loaded.'.format(model_name))

    def load_from_local(self,fp:Union[pathlib.PurePath,str,None]=None):

        """Load model from local machine."""
        if fp==None:
            self.model=MODEL_PATH.joinpath('best_model')
            self.model_tokenizer=MODEL_PATH.joinpath('best_model_t')
        else:
            if isinstance(fp,str):
                fp=pathlib.Path(fp)
            
            self.model=MODEL_PATH.joinpath(fp)
            self.model_tokenizer=MODEL_PATH.joinpath(fp)

    def prepare_data_for_BERT_train(self,pair_text_list:List[list],random_text_list:List[list],mask_prop:float=0.15):
    
        """Function to which takes list of paired and random paired text and prepares/tokenizes data for BERT training/fine tunning"""
        
        #1. Split lists
        pair_1,pair_2=self.__split_lists(pair_text_list)
        pair_labels=list(np.zeros(len(pair_1),np.int32))
        
        not_pair_1,not_pair_2=self.__split_lists(random_text_list)
        notpair_labels=list(np.ones(len(pair_1),np.int32))
        
        #2. Concatenate examples
        sentences_1=pair_1+not_pair_1
        sentences_2=pair_2+not_pair_2
        labels=pair_labels+notpair_labels
        
        #3.Tokenize 
        bert_inputs = self.model_tokenizer(sentences_1, sentences_2,return_tensors='pt',truncation=True,padding='max_length')
        
        #4. Add labels
        bert_inputs['next_sentence_label'] = torch.LongTensor([labels]).T #1. NSP fine tunning - next sentence labeling 
        bert_inputs['labels'] = bert_inputs.input_ids.detach().clone() #2. MLM fine tunning -label cloning

        #5. Add masks (for MLP head)
        bert_inputs_masked=self.__mask_inputids(bert_inputs,mask_prop)
        
        return bert_inputs_masked

    def train_BERT(self,inputs,model:Union[BertForPreTraining,None]=None,epochs:int=5,batch_size:int=16,learning_rate:float=1e-4):
        
        #1. Model init
        if model==None:
            model=self.model

        #2. Convert bert inputs (dict) returned by BertDataset class to dataset loader object
        dataset = BertDataset(inputs)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        #3. Initialize model training mode and optmizer
        model.train()
        optim = AdamW(model.parameters(), lr=learning_rate)
        
        #4. Loop over epochs acumulate losses
        loss_acum=defaultdict(list)
        for epoch in range(epochs):
            #4.1 setup loop with TQDM and dataloader
            loop = tqdm(dataset_loader, leave=True)
            for i,batch in enumerate(loop):
                #4.2 initialize calculated gradients (from prev step)
                optim.zero_grad()
                
                #4.3 pull all tensor batches required for training
                input_ids = batch['input_ids']
                token_type_ids = batch['token_type_ids']
                attention_mask = batch['attention_mask']
                next_sentence_label = batch['next_sentence_label']
                labels = batch['labels']
                
                #4.4 process
                outputs = model(input_ids, attention_mask=attention_mask,
                                token_type_ids=token_type_ids,
                                next_sentence_label=next_sentence_label,
                                labels=labels)
                #4.5 extract loss
                loss = outputs.loss # NLLloss function (negative log likelihood loss)
                #4.6 save loss
                loss_acum['batch_{}'.format(i)]=loss
                #4.7 calculate loss for every parameter that needs grad update
                loss.backward()
                #4.8 update parameters
                optim.step()
                #4.9 print relevant info to progress bar
                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix(loss=loss.item())
                
        return model,loss_acum

    def __mask_inputids(self,inputs:transformers.tokenization_utils_base.BatchEncoding,mask_prop:float=0.15):
    
        #1. create random array of floats with equal dimensions to input_ids tensor
        rand = torch.rand(inputs.input_ids.shape)
        
        #2. create mask array
        mask_arr = (rand < mask_prop) * (inputs.input_ids != 101) * \
                (inputs.input_ids != 102) * (inputs.input_ids != 0)
        
        #3. Get mask only input id indices
        selection = []
        for i in range(inputs.input_ids.shape[0]):
            selection.append(
                torch.flatten(mask_arr[i].nonzero()).tolist()
            )
        
        #4. Mask input ids with 103 token marker    
        for i in range(inputs.input_ids.shape[0]):
            inputs.input_ids[i, selection[i]] = 103
            
        return inputs

    def __split_lists(self,lst):
    
        """"Function to split list of lists into two lists"""

        l1=[]
        l2=[]
        for t in lst:
            for s in t:
                l1.append(s[0])
                l2.append(s[1])
            
        return l1,l2

    
class BertDataset(torch.utils.data.Dataset):
    
    """Data loader class required for BERT fine tunner function"""
    
    def __init__(self, encodings):
        self.encodings = encodings
    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
    def __len__(self):
        return len(self.encodings.input_ids)
            

    