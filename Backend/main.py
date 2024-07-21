from fastapi import FastAPI, HTTPException, Query
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Tuple
import uvicorn
import pandas as pd
import json
from typing import List, Dict, Any
from database import (
    create_connection,
    create_table,
    get_all_entries,
    get_random_entry,
    add_entries,
    search_tweets_by_keyword,
    get_word_counts_db,
    filter_entries_by_date
)
from preprocess import cleanup
from vader import (
    vader_init,
    vader_analyze_tweet,
    vader_analyze_dataframe
)
from similarity import (
    load_sentence_transformer_model,
    compute_df_embeddings,
    get_top_n_similar_texts
)

# Set the option to display the full content of each column
pd.set_option('display.max_colwidth', None)

# Database configuration
DATABASE = "entries.db"

# Pydantic models
class Entry(BaseModel):
    text: str
    scores: Dict[str, float]
    sentiment: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Event handler to run on application startup.
    Establishes a database connection and creates the table if it doesn't exist.
    """
    app.state.conn = await create_connection(DATABASE)

    # Creating the table should be done in another file but was done here due to time constraints
    entries_table_exists = await create_table(app.state.conn)

    app.state.model = await load_sentence_transformer_model()

    app.state.sia = vader_init()

    if not entries_table_exists:

        # Load the data
        file_path = '17616581.tweets.jl'
        data = []

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                data.append(json.loads(line))

        # Extract text and created_at for English tweets
        text_and_created = [(line['document'].get('created_at', ''), line['document'].get('text', '')) for line in data if line['document'].get('lang', '') == "en"]

        # Convert the list of dictionaries into a DataFrame
        df = pd.DataFrame(text_and_created, columns=['date', 'text'])

        # Preprocess it
        df = cleanup(df)

        # Run VADER and append scores and sentiment
        df = vader_analyze_dataframe(app.state.sia, df)

        df = compute_df_embeddings(df, app.state.model)

        # Save whole thing to db
        if 'text' not in df.columns or 'cleaned_text' not in df.columns or 'scores' not in df.columns or 'sentiment' not in df.columns or 'date' not in df.columns or 'cleaned_date' not in df.columns or 'embedding' not in df.columns:
            raise HTTPException(status_code=400, detail="DataFrame must contain 'text', 'cleaned_text', 'scores', 'sentiment', 'date', 'cleaned_date', and 'embedding' columns")
        
        entries_data = df[['text', 'cleaned_text', 'scores', 'sentiment', 'date', 'cleaned_date', 'embedding']].values.tolist()
        await add_entries(app.state.conn, entries_data)

    yield

    # Closes the database connection.
    await app.state.conn.close()

app = FastAPI(lifespan=lifespan)

@app.get("/entries/random", response_model=Entry)
async def read_random_entry():
    """
    Retrieve a random entry from the database.

    Returns:
        Entry: The retrieved random entry.

    Raises:
        HTTPException: If there are no entries in the database.
    """
    row = await get_random_entry(app.state.conn)
    if row is None:
        raise HTTPException(status_code=404, detail="No entries found")
    return Entry(text=row[1], scores=row[2], sentiment=row[3])

@app.get("/entries/", response_model=List[Entry])
async def read_entries():
    """
    Retrieve all entries from the database.

    Returns:
        List[Entry]: A list of all entries in the database.
    """
    rows = await get_all_entries(app.state.conn)
    return [Entry(text=row[1], scores=json.loads(row[3]), sentiment=row[4]) for row in rows]

@app.get("/process_string", response_model=Dict[str, Any])
async def process_string(input_string: str):
    """
    Process the input string and return text, score, and sentiment variables.

    Args:
        input_string (str): The input string to process.

    Returns:
        dict: A dictionary containing the text, score, and sentiment variables.
    """
    tweet, score, sentiment = vader_analyze_tweet(app.state.sia, input_string)

    return {"text": tweet, "score": score, "sentiment": sentiment}

@app.get("/filter_dates", response_model=List[Tuple[int, str, dict, str, str]])
async def filter_dates(start_date: str = Query(..., regex=r"^\d{8}$"), end_date: str = Query(..., regex=r"^\d{8}$")):
    """
    Filter texts by date range.

    Args:
        start_date (str): Start date in YYYYMMDD format.
        end_date (str): End date in YYYYMMDD format.

    Returns:
        List[Tuple[int, str, dict, str, str]]: List of entries within the specified date range.
    """
    try:
        filtered_entries = await filter_entries_by_date(app.state.conn, start_date, end_date)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return filtered_entries

@app.get("/search_tweets/")
async def search_tweets(keyword: str):
    """
    Search tweets by keyword using the FastAPI endpoint.

    Args:
        keyword (str): The keyword to search for in the tweets.

    Returns:
        dict: A dictionary with a list of tweets containing the keyword.
    """
    try:
        keyword_entries = await search_tweets_by_keyword(app.state.conn, keyword)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    return keyword_entries

@app.get("/word_counts/")
async def get_word_counts():
    """
    Retrieve word counts from the 'cleaned_text' column in the SQLite database and return a sorted list of words and their counts.

    Returns:
        dict: A dictionary with words as keys and their counts as values, sorted from highest to lowest count.
    """
    # Saving the number of times each word shows up should be saved to a database but was saved to a local variable due to time contraints
    try:
        sorted_word_counts = await get_word_counts_db(app.state.conn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return sorted_word_counts

@app.get("/entries_by_id/")
async def get_entries_by_id(start_id: int = Query(..., ge=1), end_id: int = Query(..., ge=1)):
    """
    Retrieve 'text' and 'sentiment' from the 'entries' table where 'id' is between and including the provided range.

    Args:
        start_id (int): The start of the ID range (inclusive).
        end_id (int): The end of the ID range (inclusive).

    Returns:
        list: A list of dictionaries with 'id', 'text', and 'sentiment' for each entry in the specified range.
    """
    async with app.state.conn.cursor() as cur:
        # Query the database for entries within the ID range
        await cur.execute("""
            SELECT id, text, sentiment
            FROM entries
            WHERE id BETWEEN ? AND ?
        """, (start_id, end_id))
        rows = await cur.fetchall()

    # Format the results as a list of dictionaries
    entries = [{"id": row[0], "text": row[1], "sentiment": row[2]} for row in rows]

    return entries

@app.get("/top_n_similar_texts/")
async def top_n_similar_texts(id: int, top_n: int = Query(..., ge=1)):
    """
    Retrieve the top n most similar texts to the text associated with the given ID.

    Args:
        id (int): The ID of the text to find similarities for.
        top_n (int): The number of top similar texts to return.

    Returns:
        list: A list of dictionaries with 'id' and 'text' for each similar entry.
    """
    async with app.state.conn.cursor() as cur:
        # Fetch the text associated with the given ID
        await cur.execute("SELECT cleaned_text, text FROM entries WHERE id = ?", (id,))
        row = await cur.fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Text not found for the given ID")

        text_to_compare = row[0]

        # Fetch all texts for similarity comparison
        await cur.execute("SELECT text, embedding FROM entries WHERE id != ?", (id,))
        all_texts = await cur.fetchall()

    # Convert to DataFrame
    df = pd.DataFrame(all_texts, columns=['text', 'embedding'])

    # Convert embedding column from JSON strings to lists
    df['embedding'] = df['embedding'].apply(json.loads)

    # Get top n similar texts
    similar_texts = get_top_n_similar_texts(df, text_to_compare, top_n, app.state.model)

    return similar_texts

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
