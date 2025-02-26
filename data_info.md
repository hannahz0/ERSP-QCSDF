MS Marco:
- Contains:
    - queries (dev), collection, qrels dev, and qrels train

    - queries: contains queries to train and test on
    - Collection: contains passages
    - qrels dev: query relevance for dev set
    - qrels train: query relevance for train set
        - qrels columns: query id, meaningless, document id, relevance


Splade and BM25:
- Trained on MSMarco dataset
- Contains: 
    - query_id, Q0 (meaningless), document_id, rank, score, R0 (meaningless)

