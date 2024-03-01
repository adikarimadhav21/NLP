"""
ChatGPT App Review Analysis Program :

This program analyzes app reviews using sentiment analysis and word clouds.
It reads reviews from a JSON file, cleans the text data, and generates an overall word cloud.
It also performs sentiment analysis per review and per word, displaying the results in pie charts
and bar charts respectively.
The sentiment analysis is done using the VADER  tool, and common positive and negative phrases are identified
and visualized.

Course Name: CMPS-5443-201 AdvTopCS:NaturalLangProc
Programmers:
Name: Madhav Adhikari
Name: Neeraj Chandragiri
Name: Nirupavardhan Lingareddygari
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import jsonlines
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') #used to download the 'punkt' package, which contains a pre-trained tokenization model
from nltk.corpus import stopwords
nltk.download('stopwords')

# Global variable to store all reviews
all_reviews = []
import json
def read_data():
    """
    Read app reviews from a JSON file and store them in the all_reviews list.
    """
    with open('apple_chatgpt.json', 'r', encoding='utf-8') as file:
        for line in file:
            obj = json.loads(line)
            all_reviews.append(obj['review'])

def clean_text(text):
    """
    Clean the text data by removing non-alphanumeric characters and converting to lowercase.
    """
    clean_text = ' '.join(filter(str.isalnum, text.lower().split()))
    return clean_text

def create_word_cloud():
    """
    Create a round-shaped word cloud from all the reviews.
    """
    all_text = ' '.join([clean_text(review) for review in all_reviews])
    overall_wordcloud = WordCloud(width=800, height=800,
                                  background_color='white',
                                  stopwords=None,
                                  min_font_size=10,
                                  prefer_horizontal=0).generate(all_text)
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(overall_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title('Overall Word Cloud')
    plt.show()


def sentiment_analysis_per_review():
    """
    Perform sentiment analysis on each review and display the results in a pie chart.
    """
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for review in all_reviews:
        cleaned_review = clean_text(review)
        sentiment_score = analyzer.polarity_scores(cleaned_review)
        sentiments.append(sentiment_score['compound'])

    # Count the number of positive,negative, neutral sentiment scores
    positive_count = sum(1 for score in sentiments if score > 0)
    negative_count = sum(1 for score in sentiments if score < 0)
    neutral_count = sum(1 for score in sentiments if score == 0)

    labels = ['Positive', 'Negative', 'Neutral']
    sizes = [positive_count, negative_count, neutral_count]
    colors = ['green', 'red', 'lightskyblue']
    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    plt.axis('equal')
    plt.title('Sentiment Analysis')
    plt.show()

def sentiment_analysis_per_word():
    """
    Perform sentiment analysis on each word and display the most common positive and negative phrases in bar charts.
    """
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    # Tokenize each review and flatten the list of tokens
    tokenized_reviews = [word_tokenize(clean_text(review)) for review in all_reviews]
    all_tokens = [token for sublist in tokenized_reviews for token in sublist]

    # Filter out stopwords from all tokens
    filtered_tokens = [token for token in all_tokens if token not in stop_words]

    # Calculate sentiment scores for each token
    sentiments_words = []
    for token in filtered_tokens:
        sentiment_word_score = analyzer.polarity_scores(token)
        sentiments_words.append(sentiment_word_score['compound'])

    # Initialize counters for positive and negative phrases
    positive_phrases = Counter()
    negative_phrases = Counter()

    # Determine the sentiment of each token and count occurrences
    for token, score in zip(filtered_tokens, sentiments_words):
        if score > 0:
            positive_phrases[token] += 1
        elif score < 0:
            negative_phrases[token] += 1

    # Get the  top 10 common positive and negative phrases
    positive_phrases = positive_phrases.most_common(10)
    negative_phrases = negative_phrases.most_common(10)

    # Unpack positive phrases and frequencies for plotting
    pos_phrases, pos_frequencies = zip(*positive_phrases)

    # Create a bar chart for most common positive phrases
    plt.figure(figsize=(10, 6))
    plt.bar(pos_phrases, pos_frequencies, color='green')
    plt.xlabel('Phrase')
    plt.ylabel('Frequency')
    plt.title('Most Common Positive Phrases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


    # Unpack negative phrases and frequencies for plotting
    neg_phrases, neg_frequencies = zip(*negative_phrases)

    # Create a bar chart for most common negative phrases
    plt.figure(figsize=(10, 6))
    plt.bar(neg_phrases, neg_frequencies, color='red')
    plt.xlabel('Phrase')
    plt.ylabel('Frequency')
    plt.title('Most Common Negative Phrases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to run the program.
    """
    read_data()
    create_word_cloud()
    sentiment_analysis_per_review()
    sentiment_analysis_per_word()

if __name__ == "__main__":
    main()
