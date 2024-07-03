import pandas as pd
import seaborn as sns
import sent2vec
import faiss
import numpy as np
from tqdm import tqdm
import os
import argparse
from transformers import AutoTokenizer, AutoModel
import torch

np.random.seed(42) 

model_path = "model/BioSentVec_PubMed_MIMICIII-bigram_d700.bin"
corpus_data = "corpus_data/"
corpus_embeddings = "corpus_embeddings/"

print ("---------------------------------------------")
print("""   _____ _____             _____ ______          _____   _____ _    _ 
  / ____|  __ \     /\    / ____|  ____|   /\   |  __ \ / ____| |  | |
 | (___ | |__) |   /  \  | (___ | |__     /  \  | |__) | |    | |__| |
  \___ \|  _  /   / /\ \  \___ \|  __|   / /\ \ |  _  /| |    |  __  |
  ____) | | \ \  / ____ \ ____) | |____ / ____ \| | \ \| |____| |  | |
 |_____/|_|  \_\/_/    \_\_____/|______/_/    \_\_|  \_\\_____|_|  |_|""")
print ("---------------------------------------------")
print()

def load_model2():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")
    return model, tokenizer


def get_embeddings(text, model, tokenizer, token_length=64):
    tokens=tokenizer(text,max_length=token_length,padding='max_length',return_tensors='pt', truncation=True)
    output=model(input_ids=tokens.input_ids,
             attention_mask=tokens.attention_mask).last_hidden_state
    return torch.mean(output,axis=1).detach().numpy()


def load_model(model_path):
    """Load the model"""
    model = sent2vec.Sent2vecModel()
    model.load_model(model_path)
    return model

def load_source_data(corpus_data):
    """Load source data"""
    txts = ["sra_ft_corpus.txt", "sra_corpus.txt", "study_corpus.txt", "submission_corpus.txt", ]
    print("Loading CSVs")
    dfs = {}
    for i in txts[:2]:
        print("Loading: ", i)
        df = pd.read_csv(f"{corpus_data}/{i}", sep='\t', names=['id', 'text'])
        df['text'] = df['text'].astype(str)
        df['id'] = df['id'].astype(str)
        dfs[i.replace(".txt", "")] = df
    return dfs

def load_embeddings_data(folder):
    """ Load the embeddings data """
    data = {}
    # SRA
    sra_ids = np.load(f"{corpus_embeddings}/sra_corpus.txt_ids.npy")
    sra_embeddings = np.load(f"{corpus_embeddings}/sra_corpus.txt_embeddings_data.npy").reshape(-1, 700)
    
    # Study
    #study_ids = np.load(f"{corpus_embeddings}/study_corpus.txt_ids.npy")
    #study_embeddings = np.load(f"{corpus_embeddings}/study_corpus.txt_embeddings_data.npy").reshape(-1, 700)
    
    # Submission
    #submission_ids = np.load(f"{corpus_embeddings}/submission_corpus.txt_ids.npy")
    #submission_embeddings = np.load(f"{corpus_embeddings}/submission_corpus.txt_embeddings_data.npy").reshape(-1, 700)
    
    # SRA FT
    sra_ft_ids = np.load(f"{corpus_embeddings}/sra_ft_corpus.txt_ids.npy")
    sra_ft_embeddings = np.load(f"{corpus_embeddings}/sra_ft_corpus.txt_embeddings_data.npy").reshape(-1, 700)
    
    
    # Experiment
    #experiment_ids = np.load(f"{corpus_embeddings}/experiment_corpus.txt_ids.npy")
    #experiment_embeddings = np.load(f"{corpus_embeddings}/experiment_corpus.txt_embeddings_data.npy").reshape(-1, 700)
    
    # Sample
    #sample_ids1 = np.load(f"{corpus_embeddings}/sample_splits/sample_split1.txt_ids.npy")
    #sample_embeddings1 = np.load(f"{corpus_embeddings}/sample_splits/sample_split1.txt_embeddings_data.npy").reshape(-1, 700)
    
    #sample_ids2 = np.load(f"{corpus_embeddings}/sample_splits/sample_split2.txt_ids.npy")
    #sample_embeddings2 = np.load(f"{corpus_embeddings}/sample_splits/sample_split2.txt_embeddings_data.npy").reshape(-1, 700)
    
    
    #sample_ids3 = np.load(f"{corpus_embeddings}/sample_splits/sample_split3.txt_ids.npy")
    #sample_embeddings3 = np.load(f"{corpus_embeddings}/sample_splits/sample_split3.txt_embeddings_data.npy").reshape(-1, 700)
    
    #sample_ids4 = np.load(f"{corpus_embeddings}/sample_splits/sample_split4.txt_ids.npy")
    #sample_embeddings4 = np.load(f"{corpus_embeddings}/sample_splits/sample_split4.txt_embeddings_data.npy").reshape(-1, 700)
    
    #sample_ids5 = np.load(f"{corpus_embeddings}/sample_splits/sample_split5.txt_ids.npy")
    #sample_embeddings5 = np.load(f"{corpus_embeddings}/sample_splits/sample_split5.txt_embeddings_data.npy").reshape(-1, 700)
    
    data['sra_corpus'] = [sra_ids, sra_embeddings]
    #data['study_corpus'] = [study_ids, study_embeddings]
    #data['submission_corpus'] = [submission_ids, submission_embeddings]
    data['sra_ft_corpus'] = [sra_ft_ids, sra_ft_embeddings]
    


    
    #data['sample_corpus'] = [sample_ids, sample_embeddings]
    #data['experiment_corpus'] = [experiment_ids, experiment_embeddings]


    
    return data
        

def load_index(embeddings_data):
    d = 700
    indexes = {}
    for index, key in enumerate(embeddings_data):
        file_name = f"{key}_index.bin"
        indexes[key] = faiss.read_index(file_name)
        print(f"Total vectors {key}: {indexes[key].ntotal}")
    return indexes

def build_index(embeddings_data):  
    """ Building FAISS index [SRA] """
    
    d = 700
    indexes = {}
    for index, key in enumerate(embeddings_data):
        indexes[key] = faiss.IndexFlatL2(d)
        print(f"[Index] {key}: Trained: {indexes[key].is_trained}")
        indexes[key].add(embeddings_data[key][1])
        file_name = f"{key}_index.bin"
        faiss.write_index(indexes[key], file_name) 
        print(f"Total vectors {key}: {indexes[key].ntotal}")
    return indexes



from rich.console import Console
console = Console()


def search(query, model, tokenizer, indexes, sids, source_data, save=False, verbose=False):
    """
        Search the query in all databases
        
        sids: edata (SIDS and Embeddings)
        indexes: Built index
        sdata: Source Data (All databases)
    """
    def compute_query(query):
        query = query.lower()
        return get_embeddings(query, model, tokenizer, token_length=64)
    qvec = compute_query(query)
    top_k = 5
    

    
    # Getting top K
    searches = {}
    for index, key in enumerate(indexes):
        print(f"Searching {key} index")
        D, I = indexes[key].search(qvec, top_k)
        I = I[0].tolist()
        # Getting first match
        if verbose:
            print("Matches: ", I)
        matched_sids = sids[key][0][I[0]]
        source_data_match = source_data[key]
        source_data_match = source_data_match[source_data_match.id == matched_sids].values
        searches[key] = source_data_match
    print()
    console.log(f"Search Query: {query}", style="bold red")
    print("-------------------\n")
    
    for index, key in enumerate(searches):
        console.log(f"{key} Matches\n", style="bold red")
        print("-----------------\n")
        for value in searches[key]:
            sid = value.tolist()[0]
            text = value.tolist()[1]
            print(f"{sid}: {text}\n")
    df = pd.DataFrame(data={"db": searches.keys(), 'data':searches.values()})
    return searches, df



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()
    #parser.add_argument('--query', type=str, required=True)
    #parser.add_argument('--download', nargs="?", type=str, required=False, default="no", const='no')
    #args = parser.parse_args()
    #query = str(args.query)
    #save = False
    download = "No"
    
    
    
    print()
    edata = load_embeddings_data(corpus_embeddings)
    sdata = load_source_data(corpus_data)
    #model = load_model(model_path)
    #model, tokenizer = load_model2()
    #indexes = build_index(edata)

    query = input("Enter query: ")
    
    print("---CONFIGURATION---")
    print(f"query: {query}")
    print(f"download: {download}")
    print("---CONFIGURATION---")
    
    if download == 'yes':
        save = True
    else:
        save = False
    
    searches, df = search(query, model, tokenizer, indexes, edata, sdata, save=save, verbose=False)
    
    if save:
        df.to_csv(f"{query}_matches.csv")
    else:
        print(f"save={save}")


#query=input("Enter Query: ")
#searches, df = search(query, indexes, edata, sdata, verbose=False)