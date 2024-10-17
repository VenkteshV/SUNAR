from dexter.retriever.sparse.SPLADE import SPLADE
from dexter.data.loaders.RetrieverDataset import RetrieverDataset
from dexter.config.constants import Split
from dexter.utils.metrics.retrieval.RetrievalMetrics import RetrievalMetrics
from dexter.utils.metrics.SimilarityMatch import CosineSimilarity as CosScore
from dexter.data.datastructures.hyperparameters.dpr import DenseHyperParams
import json

if __name__ == "__main__":

    config_instance = DenseHyperParams(query_encoder_path="naver/splade_v2_max",
                                     document_encoder_path="naver/splade_v2_max"
                                     ,batch_size=4)
    # config = config_instance.get_all_params()
    corpus_path = "data/wiki_musique_corpus.json"

    loader = RetrieverDataset("musiqueqa","wiki-musiqueqa-corpus","evaluation/config.ini",Split.TRAIN)
    queries, qrels, corpus = loader.qrels()
    print("queries",len(queries),len(qrels),len(corpus),queries[0])
    tasb_search = SPLADE(config_instance)

  

    similarity_measure = CosScore()
    response = tasb_search.retrieve(corpus,queries,1000,similarity_measure,chunk=True,chunksize=100000,data_name="wikimultihop_train")
    print("indices",len(response))
    wiki_docs = {}
    with open("mqa_splade_train.json","w") as f:
        json.dump(response,f)
    for index, key in enumerate(list(response.keys())):
        wiki_docs[key] = []
        for idx in list(response[key].keys()):
            corpus_id = int(idx)
            wiki_docs[key].append(corpus[corpus_id].text())
    with open("mqa_splade_docs_train.json","w") as f:
        json.dump(wiki_docs,f)
    metrics = RetrievalMetrics(k_values=[1,10,100,1000])
    print(metrics.evaluate_retrieval(qrels=qrels,results=response))
