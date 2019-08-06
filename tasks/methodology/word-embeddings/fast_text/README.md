# Demo of FastText Word Embeddings Methodolgy

### Intoduction
A simple demonstration of the FastText word embeddings is shown on 20 Newsgroup dataset fetched from Sckit-Learn

### Running Locally
cd tasks/methodology/word-embeddings
mlflow run . -P word_1='word_1' -P word_2='word_2'

Example:
mlflow run . -P word_1=religion -P word_2=technology

### Output
ML Flow is used to log input parameters:
* word_1
* word_2

ML Flow is used to log metrics.
* vector_size
* similarity_score between word_1 and word_2

Most similar words to word_1 and word_2 are printed on console
most_similar_to_word_1
most_similar_to_word_2
