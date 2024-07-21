import numpy as np
from sentence_transformers import SentenceTransformer
import asyncio
from sklearn.metrics.pairwise import cosine_similarity
import json

async def load_sentence_transformer_model():
    """
    Asynchronously load the all-MiniLM-L6-v2 model from SentenceTransformers.

    Returns:
    SentenceTransformer: The loaded SentenceTransformer model.
    """
    loop = asyncio.get_event_loop()
    model = await loop.run_in_executor(None, SentenceTransformer, 'all-MiniLM-L6-v2')
    return model

def compute_df_embeddings(df, model):
    """
    Compute sentence embeddings for each text in the DataFrame and append the embeddings as a new column.

    Args:
    df (pd.DataFrame): Input DataFrame containing a column named 'cleaned_text' with sentences to be embedded.
    model (SentenceTransformer): Preloaded SentenceTransformer model to compute embeddings.

    Returns:
    pd.DataFrame: DataFrame with an additional column 'embedding' containing the computed embeddings.

    Raises:
    ValueError: If the DataFrame does not contain a 'cleaned_text' column.
    """
    # Ensure the DataFrame has a 'text' column
    if 'cleaned_text' not in df.columns:
        raise ValueError("The DataFrame must contain a 'cleaned_text' column")

    # Compute embeddings for each text in the DataFrame
    #df['embedding'] = [model.encode(text, convert_to_tensor=False) for text in df['cleaned_text']]
    df['embedding'] = [json.dumps(model.encode(text, convert_to_tensor=False).tolist()) for text in df['cleaned_text']]
    #embedding = [model.encode(text, convert_to_tensor=False) for text in df['cleaned_text']]
    #df['embedding'] = [json.dumps(embed) for embed in embedding]
    #df['embedding'] = json.dumps(embedding)

    return df

def get_top_n_similar_texts(df, input_text, top_n, model):
    """
    Compute cosine similarity between the input text and the embeddings in the DataFrame.
    Return the top_n most similar texts, excluding the input text itself.

    Args:
    df (pd.DataFrame): DataFrame containing a 'text' column and a 'embedding' column.
    input_text (str): The input text to compare against.
    top_n (int): The number of top similar texts to return.
    model (SentenceTransformer): Preloaded SentenceTransformer model to compute the embedding for the input text.

    Returns:
    list: A list of the top_n most similar texts.
    """

    # Compute embedding for the input text
    input_embedding = model.encode(input_text, convert_to_tensor=False).tolist()

    # Stack all embeddings into a matrix
    embeddings_matrix = np.vstack(df['embedding'].values)
    #embeddings_matrix = np.vstack(df[1].values)

    # Compute cosine similarity
    similarities = cosine_similarity([input_embedding], embeddings_matrix).flatten()

    # Add similarity scores to DataFrame
    df['similarity'] = similarities

    # Ensure we don't return the same text as the input
    #filtered_df = df[df['cleaned_text'] != input_text]

    # Get top N most similar texts
    most_similar_texts = df.nlargest(top_n, 'similarity')['text'].tolist()

    return most_similar_texts