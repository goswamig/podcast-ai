from transformers import pipeline

def answer_question(transcript, question):
    print("Answering question...")
    nlp = pipeline('question-answering')
    context = transcript
    result = nlp(question=question, context=context)
    answer = result['answer']
    print("Answer:", answer)
    return answer

