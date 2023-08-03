import os
import nltk
import numpy as np
import pinecone
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import tensorflow_hub as hub
import tensorflow_text

def preprocess_transcript(transcript):
    sentences = nltk.sent_tokenize(transcript.lower())
    return sentences

def create_vectorizers(transcript_sentences):
    vectorizers = {
        'tfidf': TfidfVectorizer(),
        'word2vec': Word2Vec(sentences=[sentence.split() for sentence in transcript_sentences], vector_size=100, window=5, min_count=1, workers=4),
        'glove': KeyedVectors.load_word2vec_format('glove.6B.100d.w2v.txt', binary=False),
        'fasttext': Word2Vec(sentences=[sentence.split() for sentence in transcript_sentences], vector_size=100, window=5, min_count=1, workers=4),
        'doc2vec': Doc2Vec(vector_size=100, window=2, min_count=1, workers=4, epochs=100),
        'universal_sentence_encoder': hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    }

    tagged_data = [TaggedDocument(words=sentence.split(), tags=[str(i)]) for i, sentence in enumerate(transcript_sentences)]
    vectorizers['doc2vec'].build_vocab(tagged_data)
    vectorizers['doc2vec'].train(tagged_data, total_examples=len(tagged_data), epochs=vectorizers['doc2vec'].epochs)

    return vectorizers

def create_pinecone_index(transcript_sentences, vectorizers):
    pinecone.init(api_key="YOUR_PINECONE_API_KEY")

    index_name = "podcast_transcript_index"
    pinecone.create_index(index_name, dimension=100)

    segment_vectors = np.array([vectorizers["doc2vec"].infer_vector(sentence.split()) for sentence in transcript_sentences])
    pinecone.index(index_name).upsert(ids=np.arange(len(transcript_sentences)), vectors=segment_vectors)
    pinecone.index(index_name).wait_for_ready()

    return index_name

def find_most_similar_segment(question_vectors, index_name, transcript_sentences):
    pinecone.init(api_key="YOUR_PINECONE_API_KEY")

    results = pinecone.index(index_name).query(queries=question_vectors, top_k=1)
    most_similar_index = results[0][0]

    return transcript_sentences[most_similar_index]

def answer_question(transcript, question):
    # Load Pinecone API key from environment variables
    pinecone_api_key = os.environ.get("YOUR_PINECONE_API_KEY")

    if not pinecone_api_key:
        raise ValueError("Pinecone API key not found in environment variables.")

    transcript_sentences = preprocess_transcript(transcript)

    # Check if vectorizers and index already exist
    if not hasattr(answer_question, "vectorizers"):
        answer_question.vectorizers = create_vectorizers(transcript_sentences)
        answer_question.index_name = create_pinecone_index(transcript_sentences, answer_question.vectorizers)

    question_vectors = process_question(question, answer_question.vectorizers)
    most_similar_segment = find_most_similar_segment(question_vectors["doc2vec"], answer_question.index_name, transcript_sentences)

    return most_similar_segment

