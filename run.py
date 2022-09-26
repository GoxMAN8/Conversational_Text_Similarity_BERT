from QueryMatcher import *
from BERTpredictor import *

QUERY=[]

if __name__=="__main__":

    #1. Read emebeddings and initialize models
    b_predictor=BERTpredictor() # Initialize bert predictor
    b_predictor.load_from_local() # Load models from local (hard coded as best model and best tokenizer)
    q_matcher=QueryMatcher(b_predictor.model,b_predictor.model_tokenizer,load_embeddings=True,load_indexes=True)

    #2. Ask for query (input 1)
    query_0=input("What are you interested in?:")
    QUERY.append(query_0)

    #3. Maintain loop until query satisfied
    running=True
    while running==True:
        #4. Prepare query.
        scores,best_question=q_matcher.match_query(QUERY)

        #5. Print output
        print('Did you mean? : {}'.format(best_question))
        answer_0=input('Please enter Y for yes or N for no:')
        if answer_0.lower() in ['y','yes']:
            running=False
            break
        elif answer_0.lower() in ['n','no']:
            answer_1=input('Do you want to expand your query (Y/N) ?:')
            if answer_1.lower() in ['y','yes']:
                query_1=input("Please expand:")
                QUERY.append(query_1)
            else:
                answer_1=input('Do you want to start over (Y/N) ?:')
                if answer_1.lower() in ['y','yes']:
                    QUERY=[]
                    query_0=input("What are you interested in?:")
                    QUERY.append(query_0)
                else:
                    print('Your best matching question : {}.'.format(best_question))
                    break
        else:
            print('Your best matching question : {}.'.format(best_question))
            break












    