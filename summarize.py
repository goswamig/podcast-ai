import tiktoken
import openai

MODEL = "text-davinci-002"
TOKEN_LIMIT = 4096
OVERLAP_RATIO = 0.25

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("gpt2")  # Use gpt2 encoding for token counting
    num_tokens = len(encoding.encode(text))
    return num_tokens

def summarize_podcast(transcript, summary_fraction):
    with open(transcript, 'r') as file:
        text = file.read()

    words = text.split()
    instruction = "\n\nSummarize this text from a podcast. Extract the key points of the discussion between the speakers."
    max_tokens_for_output = int(summary_fraction * TOKEN_LIMIT)

    # Count tokens in instruction
    tokens_for_instruction = count_tokens(instruction)

    context_length = TOKEN_LIMIT - tokens_for_instruction  # Consider only instruction tokens for context length

    # Calculate the number of tokens for each chunk
    tokens_per_chunk = int(context_length / (1 + OVERLAP_RATIO))

    chunks, chunk = [], []
    tokens_in_chunk = 0
    for word in words:
        tokens_in_word = count_tokens(word)
        if tokens_in_chunk + tokens_in_word > tokens_per_chunk:
            chunks.append(chunk)
            chunk = []
            tokens_in_chunk = 0
        chunk.append(word)
        tokens_in_chunk += tokens_in_word
    if chunk:
        chunks.append(chunk)

    summaries = []
    for i, chunk in enumerate(chunks):
        # Add overlap with the previous chunk except for the first chunk
        if i != 0:
            overlap = " ".join(chunks[i - 1][-int(tokens_per_chunk * OVERLAP_RATIO):])
            chunk_text = " ".join(chunk)  # Convert chunk from list to string
            chunk = overlap + " " + chunk_text  # Concatenate overlap and chunk as strings
        else:
            chunk = " ".join(chunk)  # Convert first chunk from list to string

        # Check if the chunk plus instruction exceeds the maximum allowed context length
        prompt_tokens = count_tokens(chunk + instruction)
        if prompt_tokens > TOKEN_LIMIT:
            # Truncate the chunk to fit within the context length
            chunk = chunk[:context_length]

        # Adjust the max_tokens_for_output based on the remaining context length
        max_tokens_for_output = context_length - count_tokens(chunk)

        response = openai.Completion.create(
            engine=MODEL,
            prompt=chunk + instruction,
            temperature=0.2,
            max_tokens=max_tokens_for_output
        )

        summaries.append(response.choices[0].text.strip())

    print("text_sz: %d, words_sz: %d, chunks_sz:%d, summary_sz:%d" % (len(text), len(words), len(chunks), len(summaries)))
    for i in range(len(summaries)):
       print(len(summaries[i]))
    return " ".join(summaries)

# Example usage
# transcript = "path/to/transcript.txt"
# summary = summarize_podcast(transcript, 0.3)
# print("Summary:", summary)

