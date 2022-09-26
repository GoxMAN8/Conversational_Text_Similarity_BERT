from typing import Union,List
import nlpaug.augmenter.word as naw
import pathlib
import os
import nltk
import random
import pandas as pd
import numpy as np
import translators as ts
import shutil
import pickle
import ast

CORPUS_PATH=pathlib.Path().absolute().joinpath('corpus')
CORPUS_PATH_AUG=pathlib.Path().absolute().joinpath('corpus_augmented')
CORPUS_LABELLED=pathlib.Path().absolute().joinpath('corpus_labelled')

Q_PATH=pathlib.Path().absolute().joinpath('questions') # Path to questions folder


class DataPrepper():

    def __init__(self,corp_path:Union[pathlib.PurePath,str]=CORPUS_PATH,corp_aug_path:Union[pathlib.PurePath,str]=CORPUS_PATH_AUG,q_path:Union[pathlib.PurePath,str]=Q_PATH):
        
        self.corp_path=corp_path if isinstance(corp_path,pathlib.PurePath) else pathlib.Path(corp_path)
        self.corp_aug_path=corp_aug_path if isinstance(corp_aug_path,pathlib.PurePath) else pathlib.Path(corp_aug_path)
        self.q_path=q_path if isinstance(q_path,pathlib.PurePath) else pathlib.Path(q_path)

        self.data=list() # Placeholder for normal corpus
        self.data_aug=list() # Placeholder for augmented corpus
        
        self.data_index={} # Index filename to list index       

    def txt_to_lists(self,path:Union[pathlib.PurePath,str,None]=None,to_sentence:bool=True,save_index:bool=False):
    
        """Function takes path with text files as argument and returns list of lists one list per text file."""
        #0. Use defaulth corpus if none
        if path==None:
            path=self.corp_path
        
        #1. Read text int list of lists
        nltk.download('punkt')
        text=[]
        i=0
        for f in os.listdir(path):
            if f.split('.')[-1]=='txt':
                r = open(pathlib.Path(path).joinpath(f), "r",encoding='utf-8').readlines()
                #2. Breakdown into sentences (rather than lines)
                if to_sentence:
                    r=nltk.tokenize.sent_tokenize(' '.join(r))
                text.append(r)

                #3. Store file index
                self.data_index[i]={'filename':f,'text':r}
                i+=1
            else:
                print('File {} had been skipped due to unsuported extension.'.format(f))

        self.data=text

        #2. Save index
        if save_index:
            self.save_files(path.joinpath('_index'),self.data_index)

        return text

    def text_list_to_sent_pairs(self,txt_list:List[list]):
    
        """Convert list of texts into lists of subsequent sentences. (left-> first, right->subsequent). Returns lists of tuples."""
        
        sent_pairs_all=[]
        # Loop over text and create two list of tuples like [(sentence, subsequent sentence),.....]
        for t in txt_list:
            txt_list_1=t[:-1]
            txt_list_2=t[1:]
            sent_pairs_temp=[tuple((t1,t2)) for t1,t2 in zip(txt_list_1,txt_list_2)]
            sent_pairs_all.append(sent_pairs_temp)

        self.data=sent_pairs_all
                
        return sent_pairs_all


    def sent_pairs_to_random_pairs(self,text_list:List[list],text_resample_size:Union[float,int]=1.0,sent_resample_size:Union[float,int]=1.0,n_resamples:int=5):
    
        """Function to resample list of texts (output from text_list_to_sent_pairs) with subsequent sentence pairs into list of random pairs. Retruns lists of tuples"""
        
        text_list_new=[]
        #1. First step : chose random list of texts to be resampled from
        if isinstance(text_resample_size,int):
            text_list_resampled=random.sample(text_list, min(text_resample_size,len(text_list)))
        elif isinstance(text_resample_size,float):
            text_list_resampled=random.sample(text_list, min(int(text_resample_size*len(text_list)),len(text_list)))
            
        #2. Second step chose random sententces within text to be resampled from
        sentence_list_resampled_all=[]
        for t in text_list_resampled:
            if isinstance(sent_resample_size,int):
                sentence_list_resampled=random.sample(t, min(sent_resample_size,len(t)))
            elif isinstance(sent_resample_size,float):
                sentence_list_resampled=random.sample(t, min(int(sent_resample_size*len(t)),len(t)))
            
            sentence_list_resampled_all.append(sentence_list_resampled)
                
        #3. Third step to resample random pairs of sentences (flatten the original list-no more preservation of text structure)
        all_sentences=list(np.unique(np.concatenate(text_list).flat))
        for t in sentence_list_resampled_all:
            text_list_temp=[]
            for s in t:
                all_sentences_temp=all_sentences.copy()
                all_sentences_temp.remove(s[1]) # Remove next sentences
                all_sentences_temp.remove(s[0]) # Remove same sentences
                random_sent=list(np.unique(random.sample(all_sentences_temp,n_resamples))) # List of random sentences
                for r in random_sent:
                    text_list_temp.append(tuple((s[0],r))) # Create list of randomized sentence pairs
            
            text_list_new.append(text_list_temp)

        self.data=text_list_new
        
        return text_list_new

    
    def aug_syn_swap(self,text_list:List[list],aug_p:float=0.3,aug_min:int=1, aug_max:int=10,n_new_sent=2,stopwords:Union[list,None]=None,
    save:bool=False,path:Union[pathlib.PurePath,str,None]=None):
    
        "Function to augment text list of lists based on synonyms from wordnet."
        #0. Stopwords loader
        if stopwords==None:
            nltk.download('stopwords')
            stopwords=nltk.corpus.stopwords.words('english')
        
        #1. Initialize data augmenter
        aug = naw.SynonymAug(aug_src='wordnet',aug_p=aug_p,aug_min=aug_min,aug_max=aug_max,stopwords=stopwords)
        
        #2. Augment
        aug_text_list_all=[]
        #2.1Loop over text
        for t in text_list:
            aug_text_list_temp=[]
            #2.2 Loop over sentence pairs
            for s in t:
                # Created augmented synonym sentences based on wordent synonyms
                syn_t1=aug.augment(s[0],n=n_new_sent)
                syn_t2=aug.augment(s[1],n=n_new_sent)

                #2.3 Create new list
                for i,j in zip(syn_t1,syn_t2):
                    aug_text_list_temp.append(tuple((i,j)))
                    
            #3. Append to overall text list        
            aug_text_list_all.append(aug_text_list_temp)

            self.data_aug=aug_text_list_all

        #3. Store data if required
        if save:
            self.store_aug_data(aug_text_list_all,path)
            
        return aug_text_list_all

    def aug_trans_swap(self,text_list:List[list],from_lang='en',to_lang='de',
    save:bool=False,path:Union[pathlib.PurePath,str,None]=None):
    
        """Text augmentation using reverse translation via google translator."""

        #1. Loop over texts
        aug_text_all=[]
        for t in text_list:
            
            #2. Over sentence pairs within text
            aug_text_temp=[]
            for s in t:
                # Translate from english to germam and then back
                s0=ts.google(s[0], from_language=from_lang, to_language=to_lang)
                s0_aug=ts.google(s0, from_language=to_lang, to_language=from_lang)
                
                s1=ts.google(s[1], from_language=from_lang, to_language=to_lang)
                s1_aug=ts.google(s1, from_language=to_lang, to_language=from_lang)
                
                aug_text_temp.append(tuple((s0_aug,s1_aug)))
                
            aug_text_all.append(aug_text_temp)

        self.data_aug=aug_text_all

        #3. Store data if required
        if save:
            self.store_aug_data(aug_text_all,path)
        
        return aug_text_all

    def store_aug_data(self,data:List[list],folder:Union[pathlib.PurePath,str,None]=None,postfix:str='aug'):

        #1. Create directory
        if folder==None:
            folder=self.corp_aug_path
        
        if isinstance(folder,str):
            folder=pathlib.Path(folder)

        folder.mkdir(parents=False, exist_ok=True)

        #2. Loop over data and create files
        for i,d in enumerate(data):
            with open(folder.joinpath('{}_{}.txt'.format(self.data_index[i]['filename'].split('.')[0],postfix)), 'w') as f:
                for line in d:
                    f.write(line[0])
                    f.write('\n')

    def prepare_corpus_labelled(self,path:Union[pathlib.PurePath,str,None]=None):

        #0. Paths
        if path==None:
            path=CORPUS_LABELLED
        else:
            path=pathlib.Path(path)

        #1. Read raw files
        raw_qlabelled_list=self.txt_to_lists(path,to_sentence=False)

        d=pd.DataFrame(columns=['log_file','question_file','relevant_q_id'])
        
        #2. Convert dict to dataframe
        for k,v in self.data_index.items():
            i=0
            c_file=v['filename']
            
            for t in v['text']:
                t_0=t.split(' ')
                q_file=t_0[0]
                
                ids=ast.literal_eval(t_0[1])
                
                for i in ids:
                    d.loc[i,'log_file']=c_file
                    d.loc[i,'question_file']=q_file
                    d.loc[i,'relevant_q_id']=i
        
        return d


    def save_files(self,fp,f):

        """Save file with filename f to filepath fp"""
        #1. Make folder if does not exist
        fp.mkdir(parents=False, exist_ok=True)

        #2. String conversion to pathlib
        if isinstance(fp,str):
            fp=pathlib.Path(fp)
        
        #3. Dump pickle file
        with open(fp, 'wb') as handle:
            pickle.dump(f, handle)


    def load_files(self,fp,f):

        """Load file with filename f from path fp"""

        #0. Adjust for string instance
        if isinstance(fp,str):
            fp=pathlib.Path(fp)

        #1. Load pickle file
        with open(fp.joinpath(f), 'rb') as handle:
            b = pickle.load(handle)

        return b


    def clear_vars(self):

        """Function to clear class variables."""

        self.data=list() # Placeholder for normal corpus
        self.data_aug=list() # Placeholder for augmented corpus
        self.data_index={} # Index filename to list index


    def clear_folder(self,folder:Union[pathlib.PurePath,str]):
        
        #1. Forma path
        if isinstance(folder,str):
            folder=pathlib.Path(folder)

        #2. Delete folder
        if folder.is_dir():
            shutil.rmtree(folder)
        else:
            print('Folder {} does not exist.'.format(folder))
