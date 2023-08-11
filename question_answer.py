import os
import openai
import pinecone
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_TOKENS_PER_CHUNK = 2048

# Initialize APIs and keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = "my-podcast-index"

# Initialize the tokenizer
def count_tokens(text):
    encoding = tiktoken.get_encoding("gpt2")
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_text(text, max_tokens):
    words = text.split()
    chunks = []
    chunk = []
    num_tokens = 0
    for word in words:
        word_tokens = count_tokens(word)
        if num_tokens + word_tokens > max_tokens:
            chunks.append(' '.join(chunk))
            chunk = []
            num_tokens = 0
        chunk.append(word)
        num_tokens += word_tokens
    chunks.append(' '.join(chunk))
    return chunks

def initialize_pinecone_index():
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=len(embeds[0]))
        logger.info("Pinecone index '%s' created.", index_name)

def get_podcast_transcript_chunks():
    with open('/Users/gauta/Downloads/podcasts/transcripts/httpslexfridman.comgeorge-hotz-3-transcript.txt', 'r') as file:
        transcript = file.read()
    return split_text(transcript, MAX_TOKENS_PER_CHUNK)

def store_embeddings_in_pinecone(embeds, chunks):
    index = pinecone.Index(index_name=index_name)
    ids_batch = [str(n) for n in range(len(chunks))]
    meta = [{'text': line} for line in chunks]
    to_upsert = zip(ids_batch, embeds, meta)
    index.upsert(vectors=list(to_upsert))
    logger.info("Embeddings stored in Pinecone index '%s'.", index_name)

# TODO: Store the embedding and pinecone storage details 
# Avoid calling the embedding generation and storage twice
def get_or_create_embeddings():
    chunks = get_podcast_transcript_chunks()

    # Check if embeddings have already been created and stored
    if embeddings_exist_in_storage():
        logger.info("Embeddings already exist. Skipping creation and storage.")
        return load_embeddings_from_storage()

    chunk_list = chunks[:]
    res = openai.Embedding.create(input=chunk_list, engine="text-embedding-ada-002")
    embeds = [record['embedding'] for record in res['data']]
    
    initialize_pinecone_index()
    store_embeddings_in_pinecone(embeds, chunks)
    return chunks, embeds

def embeddings_exist_in_storage():
    # Implement your logic here to check if embeddings are already stored
    return False

def load_embeddings_from_storage():
    # Implement your logic here to load and return embeddings from storage
    return [], []

def get_answer_from_generative_model(question, index):
    res = openai.Embedding.create(input=[question], engine="text-embedding-ada-002")
    res = index.query([res['data'][0]['embedding']], top_k=1, include_metadata=True)
    prompt = f"Context: {res['matches'][0]['id']}\nQuestion: {question}\nAnswer:"
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
    )
    return res['choices'][0]['text'].strip()

# Main execution
def main():
    chunks, embeds = get_or_create_embeddings()

    index = pinecone.Index(index_name=index_name)

    while True:
        try:
            user_query = input("Enter your query (or type 'quit' to exit): ")
            if user_query.lower() == 'quit':
                print("Exiting...")
                break

            answer = get_answer_from_generative_model(user_query, index)
            print("Answer:", answer)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

