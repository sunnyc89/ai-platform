import sys
import string
import pandas as pd
from gensim.models import FastText
from sklearn.datasets import fetch_20newsgroups
import mlflow

def clean_text(text):
    # Tokenizing
    tokens = text.split()
    # Removing punctuation
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]
    # Removing non alphabetic tokens
    tokens = [word for word in tokens if word.isalpha() and len(word) > 1]
    return tokens


if __name__ == "__main__":
    # Reading inputs from command line
    word_1 = str(sys.argv[1]) if len(sys.argv) > 1 else "religion"
    word_2 = str(sys.argv[2]) if len(sys.argv) > 2 else "technology"

    mlflow.start_run()

    # Reading the 20NewsGroup data from sckit-learn
    newsgroups = fetch_20newsgroups(remove=("headers", "footers", "quotes"), shuffle=True)
    df = pd.DataFrame(newsgroups.data, columns=['news_doc'])

    # Cleaning all documents and tokenizing
    df['cleaned_doc'] = df.news_doc.apply(clean_text)

    # Logging the input parameters
    mlflow.log_param("word_1", word_1)
    mlflow.log_param("word_2", word_2)

    # Training the Word2Vec model using gensim
    ft = FastText(df['cleaned_doc'], min_count=3, window=5, size=300, negative=5)
    mlflow.log_metric("vector_size", ft.wv.vector_size)

    # Finding the similarity between the words
    similarity_score = ft.wv.similarity(word_1, word_2)
    print("similarity_score", similarity_score)
    mlflow.log_metric("similarity_score", similarity_score)

    # Finding the most similar words
    most_similar_to_word_1 = ft.wv.similar_by_word(word_1)
    print(most_similar_to_word_1)
    most_similar_to_word_2 = ft.wv.similar_by_word(word_2)
    print(most_similar_to_word_2)

    mlflow.end_run()
