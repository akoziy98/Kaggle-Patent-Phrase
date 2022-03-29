# Kaggle-Patent-Phrase
Repository for Kaggle competition [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/overview)

This is my own idea, how to solve task for text similarity in some context. 

1. Parse XML patents data to obtain text. This is our contexts.
2. Use simple model from sentence-transformers to vectorize this data. It shuld better use [deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) or [bert-for-patents](https://huggingface.co/anferico/bert-for-patents) from HuggingFace.
3. For each anchor and target from train dataset we make it's vectorization.
4. Make a retrieval separately for anchor and target, using [facebook faiss](https://github.com/facebookresearch/faiss) to obtain top 100 sentences from contexts.
5. Get information from this retrieval. We used number of intersections of context sentences in top 10, 30 and 100. Also we used cosine similarity for anchor and target, and whole number of the contexts.
6. Learn 5-classes classificator on the obtained features. We try to use LightGBM and custom NN for this task.
7. Make a validation with Pearson correlation

For this quite complex pipeline was obtained really poor Pearson score -- around 0.5. While simple pipeline with using only pretrained bert-for-patents with similarity between anchor and target in given contexts allows to get score about 0.8.
