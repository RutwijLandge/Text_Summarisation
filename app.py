from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from heapq import nlargest
from transformers import pipeline

app = Flask(__name__)

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load the BART summarizer
bart_summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Extractive summarization (NLTK)
def extractive_summary_nltk(text, summary_length=2):
    if not text.strip():
        return "Error: Input text is empty."

    sentences = sent_tokenize(text)
    if len(sentences) < summary_length:
        return "Error: Not enough sentences for the requested summary length."

    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words("english"))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_frequencies = nltk.FreqDist(filtered_words)

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = word_tokenize(sentence.lower())
        score = sum(word_frequencies[word] for word in sentence_words if word in word_frequencies)
        sentence_scores[sentence] = score

    top_sentences = nlargest(summary_length, sentence_scores, key=sentence_scores.get)
    return ' '.join(top_sentences)

# Abstractive summarization (BART)
def abstractive_summary_bart(text, min_length=50, max_length=150):
    if len(text.split()) < 30:
        return "Error: Text too short for abstractive summarization."

    summary = bart_summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

@app.route("/", methods=["GET", "POST"])
def index():
    summary = ""
    if request.method == "POST":
        text = request.form["text"]
        summary_type = request.form["summary_type"]

        if summary_type == "extractive_nltk":
            summary = extractive_summary_nltk(text)
        elif summary_type == "abstractive_bart":
            summary = abstractive_summary_bart(text)
        else:
            summary = "Invalid method selected!"

    return render_template("index.html", summary=summary)

if __name__ == "__main__":
    app.run(debug=True)
