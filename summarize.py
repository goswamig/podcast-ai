import tiktoken
import openai

MODEL="text-davinci-002"
TOKEN_LIMIT=4096

def count_tokens(text):
    """Returns the number of tokens in a text string."""
    # text-davinci model uses p50k_base encoding
    encoding = tiktoken.get_encoding("p50k_base")
    num_tokens = len(encoding.encode(text))
    return num_tokens

    """Returns the number of tokens in a text string."""
    encoding =  tiktoken.encoding_for_model("gpt-3.5-turbo")
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

    context_length = TOKEN_LIMIT - max_tokens_for_output - tokens_for_instruction
    summaries = []

    chunks, chunk = [], []
    tokens_in_chunk = 0
    for word in words:
        tokens_in_word = count_tokens(word)
        if tokens_in_chunk + tokens_in_word > context_length:
            chunks.append(chunk)
            chunk = []
            tokens_in_chunk = 0
        chunk.append(word)
        tokens_in_chunk += tokens_in_word
    if chunk:
        chunks.append(chunk)
    for chunk in chunks:
        response = openai.Completion.create(
            engine=MODEL,
            prompt=" ".join(chunk) + instruction,
            temperature=0.2,
            max_tokens=max_tokens_for_output
        )

        summaries.append(response.choices[0].text.strip())
    print("text_sz: %d, words_sz: %d, chunks_sz:%d, summary_sz:%d" % (len(text), len(words), len(chunks), len(summaries)))
    return " ".join(summaries)


"""
To build final summary
- Extrat bullet points
- Improve quality: remove duplicate key points, re-arrange senteces or key points for better readability
"""   
###    # Final summary
###    print("Summarizing into overall summary")
###    response = openai.ChatCompletion.create(
###        model=GPT_MODEL,
###        messages=[
###            {
###                "role": "user",
###                "content": f"""Write a summary collated from this collection of key points extracted from an podcast summary.
###                        The summary should highlight the core argument, conclusions and evidence, and answer the user's query.
###                        User query: {query}
###                        The summary should be structured in bulleted lists following the headings Core Argument, Evidence, and Conclusions.
###                        Key points:\n{results}\nSummary:\n""",
###            }
###        ],
###        temperature=0,
###    )
###    return response




