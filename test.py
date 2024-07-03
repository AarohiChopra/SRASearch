from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")

model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract")


def get_embeddings(text,token_length):
    tokens=tokenizer(text,max_length=token_length,padding='max_length',return_tensors='pt', truncation=True)
    output=model(input_ids=tokens.input_ids,
             attention_mask=tokens.attention_mask).last_hidden_state
    return torch.mean(output,axis=1).detach().numpy().shape



print(get_embeddings("park1 is bad", token_length=64))

    