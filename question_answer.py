import os
import openai
import pinecone
import tiktoken
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
openai.Completion.create with `text-davinci-003` can have total 4096 tokens.
My output of the above call has 400 tokens, so total prompt limit is 3696.
Out of which I have ~100 tokens for start/end of prompt. Hence max chunks 
can be approx 4096 - 400 - 100i = 3596. To keep margin I will keep it 3500
"""
# Constants
MAX_TOKENS_PER_CHUNK = 3500

# Initialize APIs and keys
openai.api_key = os.getenv('OPENAI_API_KEY')
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))
index_name = "my-podcast-index"

# Initialize the tokenizer
def count_tokens(text):
    encoding = tiktoken.get_encoding("cl100k_base")
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
    #logger.info("Embeddings stored in Pinecone index '%s'.", index_name)

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

def get_answer_from_generative_model(query, index):
    query_with_contexts = retrieve(query, index)
    return complete(query_with_contexts)



def retrieve(query, index):
    res = openai.Embedding.create(
        input=[query],
        engine="text-embedding-ada-002"
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # Todo use more than one context
    # get relevant contexts
    res = index.query(xq, top_k=1, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]
    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on below context.\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\nQuestion: {query}\nAnswer:"
    )
    prompt = (
                prompt_start +
                "\n---\n".join(contexts) +
                prompt_end
            )
    # TODO: you need to make sure this final prompt, which includes user query 
     #      is smaller than 4097 (complete.create API limit) - 400(reserved for output) 
    return prompt


def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()


# Main execution
def main():
    chunks, embeds = get_or_create_embeddings()

    index = pinecone.Index(index_name=index_name)
    print("Enter your query (or type 'quit' to exit): ")
    while True:
        try:
            user_query = input("Q: ")
            if user_query.lower() == 'quit':
                print("Exiting...")
                break

            answer = get_answer_from_generative_model(user_query, index)
            print("A:", answer)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()

