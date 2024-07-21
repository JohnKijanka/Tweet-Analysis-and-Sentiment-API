import json
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from preprocess import cleanup_single

def vader_analyze_dataframe(sia, df):
    """
    Uses VADER to determine scores/sentiment and appends them to the dataframe.

    Args:
        sia (SentimentIntensityAnalyzer): VADER SentimentIntensityAnalyzer with custom words added.
        df (DataFrame): DataFrame of tweets to be analyzed

    Returns:
        DataFrame: Original DataFrame with scores/sentiment appended to the end.
    """
    # Analyze sentiment for each tweet
    df['scores'] = df['text'].apply(lambda text: sia.polarity_scores(text))
    df['sentiment'] = df['scores'].apply(lambda score: classify_sentiment(score))

    df['scores'] = df['scores'].apply(json.dumps)  # Convert scores to JSON string

    return df

def vader_analyze_tweet(sia, tweet):
    """
    Uses VADER to determine scores/sentiment of a single tweet.

    Args:
        sia (SentimentIntensityAnalyzer): VADER SentimentIntensityAnalyzer with custom words added.
        tweet (str): The tweet being analyzed.

    Returns:
        score: Score produced by the tweet.
        sentiment: Overall sentiment of the tweet.
    """
    # Analyze sentiment for each tweet
    cleaned_tweet = cleanup_single(tweet)
    score = sia.polarity_scores(cleaned_tweet)
    sentiment = classify_sentiment(score)

    return tweet, score, sentiment

def vader_init():
    """
    Initializes VADER SentimentIntensityAnalyzer and adds a list of custom words to the lexicon.

    Args:
        None

    Returns:
        SentimentIntensityAnalyzer: Sentiment Intensity Analyzer that can give a sentiment intensity score to sentences.
    """
    # Initialize VADER SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Define your custom words and sentiments
    custom_words = {
        'lit': {'neg': 0.0, 'neu': 0.1, 'pos': 0.9, 'compound': 0.9},  # Exciting, excellent
        'fire': {'neg': 0.0, 'neu': 0.1, 'pos': 0.9, 'compound': 0.9},  # Great, amazing
        'awesome': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.9},  # Fantastic, very good
        'cool': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Impressive, stylish
        'vibes': {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.0},  # General feeling or atmosphere (neutral)
        'savage': {'neg': 0.1, 'neu': 0.3, 'pos': 0.6, 'compound': 0.5},  # Bold, unapologetic, sometimes harsh
        'cringe': {'neg': 0.8, 'neu': 0.2, 'pos': 0.0, 'compound': -0.7},  # Embarrassing, awkward
        'dope': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Great, cool
        'hyped': {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.7},  # Excited, enthusiastic
        'OMG': {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.5},  # Oh my God, used for excitement or surprise
        'YAS': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Expression of approval or excitement
        'slay': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.9},  # To do something exceptionally well
        'mood': {'neg': 0.0, 'neu': 0.5, 'pos': 0.5, 'compound': 0.0},  # General feeling or state (neutral)
        'fomo': {'neg': 0.4, 'neu': 0.4, 'pos': 0.2, 'compound': -0.2},  # Fear of missing out, can be anxiety-inducing
        'sus': {'neg': 0.4, 'neu': 0.3, 'pos': 0.3, 'compound': -0.2},  # Suspicious, untrustworthy
        'BFF': {'neg': 0.0, 'neu': 0.1, 'pos': 0.9, 'compound': 0.8},  # Best friend forever, positive and affectionate
        'YOLO': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.6},  # You only live once, encouragement for bold actions
        'tbh': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.5},  # To be honest, straightforward and honest
        'shook': {'neg': 0.5, 'neu': 0.4, 'pos': 0.1, 'compound': -0.3},  # Surprised or disturbed by something
        'thirsty': {'neg': 0.7, 'neu': 0.2, 'pos': 0.1, 'compound': -0.5},  # Overly eager or desperate
        'iconic': {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.6},  # Famous, well-known for being outstanding
        'queen': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Admiration or praise, often for a person or thing
        'trash': {'neg': 0.9, 'neu': 0.1, 'pos': 0.0, 'compound': -0.8},  # Bad, low quality
        'epic': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.7},  # Great, impressive
        'fleek': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.7},  # Perfectly done, stylish
        'sorry': {'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.5},  # Apology, often used to express regret
        'basic': {'neg': 0.5, 'neu': 0.4, 'pos': 0.1, 'compound': -0.3},  # Unoriginal, mainstream (negative)
        'yass': {'neg': 0.0, 'neu': 0.1, 'pos': 0.9, 'compound': 0.8},  # Excited approval, similar to "YAS"
        'shook': {'neg': 0.5, 'neu': 0.4, 'pos': 0.1, 'compound': -0.3},  # Surprised or disturbed
        'good vibes': {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.6},  # Positive, pleasant feelings
        'no cap': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.7},  # Honest or truthful
        'lame': {'neg': 0.8, 'neu': 0.2, 'pos': 0.0, 'compound': -0.6},  # Unimpressive or dull
        'dank': {'neg': 0.0, 'neu': 0.3, 'pos': 0.7, 'compound': 0.6},  # High quality, often used for memes or humor
        'chill': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.5},  # Relaxed, laid-back
        'salty': {'neg': 0.6, 'neu': 0.3, 'pos': 0.1, 'compound': -0.4},  # Upset or bitter
        'blessed': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Feeling grateful or fortunate
        'fake': {'neg': 0.8, 'neu': 0.2, 'pos': 0.0, 'compound': -0.6},  # Not genuine, deceptive
        'sick': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Impressive, cool
        'sorry': {'neg': 0.6, 'neu': 0.4, 'pos': 0.0, 'compound': -0.5},  # Apology, often used to express regret
        'woke': {'neg': 0.0, 'neu': 0.4, 'pos': 0.6, 'compound': 0.5},  # Socially aware or conscious
        'LOL': {'neg': 0.0, 'neu': 0.1, 'pos': 0.9, 'compound': 0.7},  # Laughing out loud, humor
        'LMAO': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Laughing my ass off, amusement
        'SMH': {'neg': 0.7, 'neu': 0.2, 'pos': 0.0, 'compound': -0.6},  # Shaking my head, disappointment
        'ROFL': {'neg': 0.0, 'neu': 0.2, 'pos': 0.8, 'compound': 0.8},  # Rolling on the floor laughing, humor
    }

    # Add custom words to VADER lexicon
    for word, sentiment in custom_words.items():
        # VADER uses the lexicon to modify the sentiment score
        sia.lexicon[word] = sentiment['compound']

    return sia

# Function to classify sentiment
def classify_sentiment(score):
    if score['compound'] >= 0.05:
        return "positive"
    elif score['compound'] <= -0.05:
        return "negative"
    else:
        return "neutral"