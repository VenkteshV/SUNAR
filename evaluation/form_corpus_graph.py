import json
import torch
import json
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.data.datastructures.question import Question
from dexter.retriever.sparse.SPLADE import SPLADE
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
import json
from dexter.config.constants import Split
from dexter.retriever.dense.ColBERT.colbert.infra.config.config import ColBERTConfig
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import DotScore
from dexter.retriever.dense.TCTColBERT import TCTColBERT
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from torch import Tensor
from typing import List,Dict

def get_top_k_similar_instances(
    sentence: str, data_emb: Tensor, data: List[Dict],
    k: int, threshold: float
) -> List[Dict]:
    """get top k neighbours for a sentence.

    Args:
        sentence (str): input
        data_emb (Tensor): corpus embeddings
        data (List[Dict]): corpus
        k (int): top_k to return
        threshold (float):

    Returns:
        List[Dict]: list of top_k data points
    """
    sent_emb = sent_model.encode(sentence)
    # data_emb = self.get_embeddings_for_data(transfer_questions)
    print("new_emb", sent_emb.shape, data_emb.shape)
    text_sims = cosine_similarity(data_emb, [sent_emb]).tolist()
    results_sims = zip(range(len(text_sims)), text_sims)
    sorted_similarities = sorted(
        results_sims, key=lambda x: x[1], reverse=True)
    print("text_sims", sorted_similarities[:2])
    top_questions = []
    for idx, item in sorted_similarities[:k]:
        if item[0] > threshold:
            top_questions.append(list(data)[idx])
    return top_questions

sent_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2",device="cpu")
# Load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained("colbert-ir/colbertv2.0")
model = AutoModel.from_pretrained("colbert-ir/colbertv2.0")
model.cuda()
model.eval()
def maxsim(query_embedding, document_embedding):
    # Expand dimensions for broadcasting
    # Query: [batch_size, query_length, embedding_size] -> [batch_size, query_length, 1, embedding_size]
    # Document: [batch_size, doc_length, embedding_size] -> [batch_size, 1, doc_length, embedding_size]
    expanded_query = query_embedding.unsqueeze(2)
    expanded_doc = document_embedding.unsqueeze(1)

    # Compute cosine similarity across the embedding dimension
    sim_matrix = torch.nn.functional.cosine_similarity(expanded_query, expanded_doc, dim=-1)

    # Take the maximum similarity for each query token (across all document tokens)
    # sim_matrix shape: [batch_size, query_length, doc_length]
    max_sim_scores, _ = torch.max(sim_matrix, dim=2)

    # Average these maximum scores across all query tokens
    avg_max_sim = torch.mean(max_sim_scores, dim=1)
    return avg_max_sim

# Encode the query


if __name__ == "__main__":
    with open("wqa_splade_docs.json","r") as f:
        splade_docs = json.load(f)
    with open("wqa_splade.json","r") as f:
        response = json.load(f)
    loader = RetrieverDataset("wikimultihopqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.DEV,tokenizer=None)
    queries, qrels, corpus = loader.qrels()
    print(qrels,queries)
    config_instance = DenseHyperParams(query_encoder_path="naver/splade_v2_max",
                                     document_encoder_path="naver/splade_v2_max"
                                     ,batch_size=4)
    tasb_search = SPLADE(config_instance)


    similarity_measure = CosScore()
    #response = tasb_search.retrieve(corpus,queries,100,similarity_measure,chunk=True,chunksize=180000,data_name="wikimultihop")
    metrics = RetrievalMetrics(k_values=[1,10,50,1000])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
    print("indices",len(response))
    queries_new = []
    for index, doc in enumerate(list(corpus)):
        new_corpus = doc.title() +"[SEP]"+doc.text()
        #print(queries[index].text()+new_corpus)
        queries_new.append(Question(text=new_corpus,idx=doc.id()))

    corpus_data = []
    config_instance = ColBERTConfig(doc_maxlen=256, nbits=2, kmeans_niters=4,bsize=4, gpus=0)

    tasb_search = TCTColBERT(config_instance,checkpoint="colbert-ir/colbertv2.0")
    similarity_measure = DotScore()
    response = tasb_search.retrieve(corpus,queries_new,100)
    metrics = RetrievalMetrics(k_values=[1,10,50,100])
    #print(response)
    with open("gar_graph.json","w") as f:
        json.dump(response,f)
    wiki_docs = {}

    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for idx in list(response[key].keys()):
            corpus_id = int(idx)
            wiki_docs[key].append(corpus[corpus_id].text())
    with open("gar_graph_docs.json","w") as f:
        json.dump(wiki_docs,f)
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
