import torch
from transformers import AutoModel, AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
from transformers.modeling_outputs import BaseModelOutput

def l2_normalize(x: torch.Tensor):
    return torch.nn.functional.normalize(x, p=2, dim=-1)

def encode_query(tokenizer: PreTrainedTokenizerFast, query: str) -> BatchEncoding:
    return tokenizer(query,
                     max_length=32,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

def encode_passage(tokenizer: PreTrainedTokenizerFast, passage: str, title: str = '-') -> BatchEncoding:
    return tokenizer(title,
                     text_pair=passage,
                     max_length=144,
                     padding=True,
                     truncation=True,
                     return_tensors='pt')

tokenizer = AutoTokenizer.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
model = AutoModel.from_pretrained('intfloat/simlm-base-msmarco-finetuned')
model.eval()

with torch.no_grad():
    query_batch_dict = encode_query(tokenizer, 'what is qa')
    outputs: BaseModelOutput = model(**query_batch_dict, return_dict=True)
    query_embedding = l2_normalize(outputs.last_hidden_state[0, 0, :])

    psg1 = 'Quality assurance (QA) is a process-centered approach to ensuring that a company or organization is providing the best possible products or services. It is related to quality control, which focuses on the end result, such as testing a sample of items from a batch after production.'
    psg1_batch_dict = encode_passage(tokenizer, psg1)
    outputs: BaseModelOutput = model(**psg1_batch_dict, return_dict=True)
    psg1_embedding = l2_normalize(outputs.last_hidden_state[0, 0, :])

    psg2 = 'The Super Bowl is typically four hours long. The game itself takes about three and a half hours, with a 30 minute halftime show built in.'
    psg2_batch_dict = encode_passage(tokenizer, psg2)
    outputs: BaseModelOutput = model(**psg2_batch_dict, return_dict=True)
    psg2_embedding = l2_normalize(outputs.last_hidden_state[0, 0, :])

    # Higher cosine similarity means they are more relevant
    print("SIMLM Embedding Scripts:")
    
    #print(query_embedding.dot(psg1_embedding), query_embedding.dot(psg2_embedding))
    print("Query vs Passage 1 Similarity:",query_embedding.dot(psg1_embedding))
    print("Query vs Passage 2 Similarity:",query_embedding.dot(psg2_embedding))

    # Print embeddings
    # print("Query Embedding:", query_embedding)
    # print("Passage 1 Embedding:", psg1_embedding)
    # print("Passage 2 Embedding:", psg2_embedding)
