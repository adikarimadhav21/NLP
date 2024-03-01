from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import jsonlines
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('stopwords')


all_reviews = []

def read_data():
    # Load the reviews from the ndjson file
    with jsonlines.open('apple_chatgpt.json') as reader:
        for obj in reader:
            all_reviews.append(obj['review'])

# Function to clean text data
def clean_text(text):
    # Remove non-alphanumeric characters and convert to lowercase
    clean_text = ' '.join(filter(str.isalnum, text.lower().split()))
    return clean_text
def create_word_cloud():
    # Create overall word cloud
    all_text = ' '.join([clean_text(review) for review in all_reviews])
    overall_wordcloud = WordCloud(width=800, height=800,
                                  background_color='white',
                                  stopwords=None,
                                  min_font_size=10).generate(all_text)
    # Plot overall word cloud
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(overall_wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.title('Overall Word Cloud')
    plt.show()

def sentiment_analysis_per_review():
    # Perform sentiment analysis using VADER
    analyzer = SentimentIntensityAnalyzer()
    sentiments = []
    for review in all_reviews:
        cleaned_review = clean_text(review)
        sentiment_score = analyzer.polarity_scores(cleaned_review)
        sentiments.append(sentiment_score['compound'])

    # Sentiment analysis - Pie chart
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
    analyzer = SentimentIntensityAnalyzer()
    stop_words = set(stopwords.words('english'))
    tokenized_reviews = [word_tokenize(clean_text(review)) for review in all_reviews]

    all_tokens = [token for sublist in tokenized_reviews for token in sublist]

    # Filter out stopwords from tokenized reviews
    filtered_tokens = [token for token in all_tokens if token not in stop_words]

    sentiments_words = []
    for token in filtered_tokens:
        sentiment_word_score = analyzer.polarity_scores(token)
        sentiments_words.append(sentiment_word_score['compound'])

    positive_phrases = Counter()
    negative_phrases = Counter()

    for token, score in zip(filtered_tokens, sentiments_words):
        if score > 0:
            positive_phrases[token] += 1
        elif score < 0:
            negative_phrases[token] += 1

    positive_phrases = positive_phrases.most_common(10)
    negative_phrases = negative_phrases.most_common(10)

    pos_phrases, pos_frequencies = zip(*positive_phrases)
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(pos_phrases, pos_frequencies, color='green')
    plt.xlabel('Phrase')
    plt.ylabel('Frequency')
    plt.title('Most Common Positive Phrases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Display the chart
    plt.show()

    neg_phrases, neg_frequencies = zip(*negative_phrases)
    # Create a bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(neg_phrases, neg_frequencies, color='red')
    plt.xlabel('Phrase')
    plt.ylabel('Frequency')
    plt.title('Most Common Negative Phrases')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    # Display the chart
    plt.show()


def main():
    read_data()
    create_word_cloud()
    sentiment_analysis_per_review()
    sentiment_analysis_per_word()


if __name__ == "__main__":
    main()









