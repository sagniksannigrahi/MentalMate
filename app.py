from flask import Flask, render_template, request
from transformers import pipeline
import csv
import datetime
import os
import matplotlib.pyplot as plt


sentiment_model = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

app = Flask(__name__)

DATA_FILE = 'mood_data.csv'

def analyze_sentiment(text):
    result = sentiment_model(text)[0]
    label = result['label']  # Example: '4 stars'
    score = int(label[0])    # Extracts the number from '4 stars'
    # Map 1-5 star rating to -1.0 to +1.0 mood scale
    normalized_score = (score - 3) / 2
    return round(normalized_score, 2)


def save_mood(score):
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([datetime.date.today(), score])

@app.route("/", methods=["GET", "POST"])
def index():
    mood_score = None
    quote = None
    if request.method == "POST":
        text = request.form["journal"]
        mood_score = analyze_sentiment(text)
        save_mood(mood_score)
        generate_static_mood_graph()  # ← Add this line here


        # Motivational quotes
        if mood_score > 0.3:
            quote = "Keep shining! ☀️"
        elif mood_score < -0.3:
            quote = "It's okay to not be okay 💙"
        else:
            quote = "You're doing fine, keep going 💪"

    return render_template("index.html", mood_score=mood_score, quote=quote)

@app.route("/graph")
def mood_graph():
    if not os.path.exists(DATA_FILE):
        return "No data yet."

    dates, scores = [], []
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dates.append(row[0])
            scores.append(float(row[1]))

    plt.figure(figsize=(7, 4))
    plt.plot(dates, scores, marker='o')
    plt.xticks(rotation=45)
    plt.xlabel("Date")
    plt.ylabel("Mood Score")
    plt.title("Mood Tracker")
    plt.tight_layout()
    plt.savefig('static/mood_graph.png')
    plt.close()

    return render_template("graph.html")

def generate_static_mood_graph():
    dates, scores = [], []
    with open(DATA_FILE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            dates.append(row[0])
            scores.append(float(row[1]))

    plt.figure(figsize=(10, 5))
    plt.plot(dates, scores, marker='o', linestyle='-', color='royalblue')
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Mood Score')
    plt.title('Mood Tracker')
    plt.tight_layout()
    
    plt.savefig('static/mood_graph.png')  # Overwrites old graph
    plt.close()


    
if __name__ == "__main__":
    app.run(debug=True)

    
