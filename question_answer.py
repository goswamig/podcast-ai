import os
import openai
import pinecone
import tiktoken


# Initialize the tokenizer

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("gpt2")  # Use gpt2 encoding for token counting
    num_tokens = len(encoding.encode(text))
    return num_tokens

def split_text(text, max_tokens):
    """Splits the text into chunks that have fewer than max_tokens each."""
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
    chunks.append(' '.join(chunk))  # Don't forget the last chunk
    return chunks


# Load your podcast transcript data from a file
with open('/Users/gauta/Downloads/podcasts/transcripts/httpslexfridman.comgeorge-hotz-3-transcript.txt', 'r') as file:
    transcript = file.read()

# Split the transcript into chunks of 2048 tokens each
chunks = split_text(transcript, 2048)

# Initialize OpenAI and Pinecone
openai.api_key = os.getenv('OPENAI_API_KEY')

#Lets get all embedding in one shot 
chunk_list = []
for i in range(len(chunks)):
  chunk_list.append(chunks[i])

# Create embeddings for each chunk and store them in Pinecone
res = openai.Embedding.create(input=chunk_list, engine="text-embedding-ada-002")
embeds = [record['embedding'] for record in res['data']]

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

index_name="my-podcast-index"
# check if 'openai' index already exists (only create index if not)
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=len(embeds[0]))

index = pinecone.Index(index_name=index_name)

ids_batch = [str(n) for n in range(len(chunks)]
meta = [{'text': line} for line in chunks]
to_upsert = zip(ids_batch, embeds, meta)
index.upsert(vectors=list(to_upsert))
##Lets get all embedding in one shot 
#chunk_list = []
#for i in range(len(chunks)):
#  chunk_list.append(chunks[i])
#
## Create embeddings for each chunk and store them in Pinecone
#res = openai.Embedding.create(input=chunk_list, engine="text-embedding-ada-002")

#for i, chunk in enumerate(chunks):
#    res = openai.Embedding.create(
#        input=[chunk],
#        engine="text-embedding-ada-002"
#    )
#    print(type(res['data'][0]['embedding']))
#    #vector = pinecone.Vector(res['data'][0]['embedding'], id=f"podcast_chunk_{i}")
#    #vector = pinecone.Vector(res['data'][0]['embedding'], values=True)
#    vectors = {f"podcast_chunk_{i}": res['data'][0]['embedding'] }
#    index.upsert(vectors=vectors)
#    break

def answer_question(question):
    # Create an embedding for the question
    res = openai.Embedding.create(
        input=[question],
        engine="text-embedding-ada-002"
    )

    # Retrieve the most similar embeddings from Pinecone
    res = index.query([res['data'][0]['embedding'], top_k=1, include_metadata=True)
    
    # Construct the prompt for the generative model
    prompt = f"Context: {res['matches'][0]['id']}\nQuestion: {question}\nAnswer:"

    # Generate an answer to the question
    res = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
    )

    return res['choices'][0]['text'].strip()

# Example usage
print(answer_question("What are the main topics of the podcast?"))

