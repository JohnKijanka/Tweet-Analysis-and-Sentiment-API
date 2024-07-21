import re
from datetime import datetime

def cleanup(df):
    """
    Applies all preprocess functions to dataframe.

    Args:
        df (DataFrame): DataFrame containing tweets.

    Returns:
        DataFrame: The entire DataFrame with an appended 'cleaned_text' column after being cleaned.
    """
    # Apply text preprocessing: expand acronyms
    df['cleaned_text'] = df['text'].apply(expand_acronyms)

    # Remove any word that starts with @
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'@\w+', '', regex=True).str.strip()

    # Remove URLs from text
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'http[s]?://\S+', '', regex=True).str.strip()

    # Remove 'RT : ' from beginning of text
    df['cleaned_text'] = df['cleaned_text'].apply(lambda x: x[len('RT : '):] if x.startswith('RT : ') else x)

    # Remove '#' from the beginning of any word in the text
    df['cleaned_text'] = df['cleaned_text'].str.replace(r'\b#', '', regex=True)

    # Remove '\n' from text
    df['cleaned_text'] = df['cleaned_text'].str.replace('\n', ' ', regex=False)

    df = clean_date_column(df)

    return df

def cleanup_single(tweet):
    """
    Applies all preprocess functions to a single tweet.

    Args:
        tweet (str): String for a single tweet.

    Returns:
        str: The original tweet after being cleaned.
    """
    # Remove any word that starts with '@'
    cleaned_tweet = re.sub(r'@\w+', '', tweet)
    
    # Remove URLs
    cleaned_tweet = re.sub(r'http[s]?://\S+', '', cleaned_tweet)
    
    # Remove 'RT : ' from the beginning of the text
    if cleaned_tweet.startswith('RT : '):
        cleaned_tweet = cleaned_tweet[len('RT : '):]
    
    # Remove '#' from the beginning of any word
    cleaned_tweet = re.sub(r'\b#', '', cleaned_tweet)
    
    # Replace '\n' with a space
    cleaned_tweet = cleaned_tweet.replace('\n', ' ')
    
    # Remove any extra spaces that might be left after cleaning
    cleaned_tweet = re.sub(r'\s+', ' ', cleaned_tweet).strip()

    return cleaned_tweet

def clean_date_column(df):
    """
    Append a 'cleaned_date' column to the DataFrame with dates in the format YYYYMMDD.

    Args:
    df (pd.DataFrame): DataFrame with a 'date' column in the format 'Mon Jan 22 22:01:10 +0000 2018'.

    Returns:
    pd.DataFrame: DataFrame with an additional 'cleaned_date' column.
    """
    df['cleaned_date'] = df['date'].apply(lambda x: datetime.strptime(x, '%a %b %d %H:%M:%S %z %Y').strftime('%Y%m%d'))
    return df

# Function to expand acronyms
def expand_acronyms(text):
    acronyms = {
        'BRB': 'be right back',
        'IDK': 'I don’t know',
        'BTW': 'by the way',
        'TTYL': 'talk to you later',
        'IMO': 'in my opinion',
        'IMHO': 'in my humble opinion',
        'FYI': 'for your information',
        'TMI': 'too much information',
        'BTW': 'by the way',
        'BYOB': 'bring your own beer'
        # Add more acronyms as needed
    }
    pattern = re.compile('|'.join(acronyms.keys()), re.IGNORECASE)
    return pattern.sub(lambda x: acronyms[x.group().upper()], text)