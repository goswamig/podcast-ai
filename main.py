import os
import openai
import argparse
from transcribe import transcribe_podcast
from summarize import summarize_podcast
from question_answer import answer_question

# Set up the command-line arguments parser
parser = argparse.ArgumentParser(description='Summarize a podcast.')
parser.add_argument('--transcript', type=str, default="/Users/gauta/Downloads/podcasts/transcripts/httpslexfridman.comgeorge-hotz-3-transcript.txt", help='Path to the transcript file')
parser.add_argument('--summary_fraction', type=float, default=0.15, help='Fraction of the original text size for the summary')

args = parser.parse_args()

# Set OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Summarize the podcast
#summary = summarize_podcast(args.transcript, args.summary_fraction)
#print("Summary:", summary)
#
# Call the answer_question function with the specified question
question = "Which all company George has?"
answer = answer_question(summary, question)
print(f"Answer to '{question}': {answer}")

