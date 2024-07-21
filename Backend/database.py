import aiosqlite
from sqlite3 import Error
import random
import json
from datetime import datetime
from typing import List, Tuple
import re
from collections import Counter

async def create_connection(db_file):
    """
    Create a database connection to the SQLite database specified by db_file.

    Args:
        db_file (str): Path to the SQLite database file.

    Returns:
        aiosqlite.Connection: The database connection object if successful, None otherwise.
    """
    conn = None
    try:
        conn = await aiosqlite.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return conn

async def create_table(conn):
    """
    Create the entries table if it does not exist.

    Args:
        conn (aiosqlite.Connection): The database connection.

    Returns:
        bool: True if the table already existed, False otherwise.
    """
    table_existed = False

    async with conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='entries'") as cursor:
        if await cursor.fetchone():
            table_existed = True
        else:
            await conn.execute('''
            CREATE TABLE entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                cleaned_text TEXT NOT NULL,
                scores TEXT NOT NULL,
                sentiment TEXT NOT NULL,
                date TEXT NOT NULL,
                cleaned_date TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            ''')
            await conn.commit()

    return table_existed

async def get_all_entries(conn):
    """
    Retrieve all entries from the entries table.

    Args:
        conn (aiosqlite.Connection): The database connection object.

    Returns:
        list: A list of tuples representing all rows in the entries table.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT * FROM entries")
        rows = await cur.fetchall()
        return rows

async def get_random_entry(conn):
    """
    Retrieve a random entry from the database.

    Args:
        conn (aiosqlite.Connection): The database connection.

    Returns:
        tuple: The retrieved entry as a tuple (id, text, scores, sentiment) or None if no entry is found.
    """
    async with conn.execute("SELECT COUNT(*) FROM entries") as cursor:
        result = await cursor.fetchone()
        count = result[0]
        if count == 0:
            return None
        random_id = random.randint(1, count)
    
    async with conn.execute("SELECT id, text, scores, sentiment FROM entries WHERE id = ?", (random_id,)) as cursor:
        entry = await cursor.fetchone()
        if entry:
            # Convert scores back from JSON string to a dictionary
            entry = (entry[0], entry[1], json.loads(entry[2]), entry[3])
        return entry
    
async def add_entries(conn, entries):
    """
    Add multiple entries to the entries table.

    Args:
        conn (aiosqlite.Connection): The database connection object.
        entries (list): A list of tuples containing the text, cleaned_text, scores, sentiment, date, cleaned_date, and embedding of each entry.

    Returns:
        list: A list of IDs of the last inserted rows.
    """
    sql = '''INSERT INTO entries(text, cleaned_text, scores, sentiment, date, cleaned_date, embedding)
             VALUES(?, ?, ?, ?, ?, ?, ?)'''
    async with conn.cursor() as cur:
        ids = []
        for entry in entries:
            await cur.execute(sql, entry)
            ids.append(cur.lastrowid)
        await conn.commit()
        return ids
    
'text', 'cleaned_text', 'scores', 'sentiment', 'date', 'cleaned_date'
    
async def filter_entries_by_date(conn, start_date: str, end_date: str) -> List[Tuple[int, str, dict, str]]:
    """
    Filter entries by date range.

    Args:
        conn (aiosqlite.Connection): The database connection.
        start_date (str): The start date in the format YYYYMMDD.
        end_date (str): The end date in the format YYYYMMDD.

    Returns:
        list: A list of tuples representing the filtered rows (id, text, scores, sentiment, date).
    """
    try:
        start_date_dt = datetime.strptime(start_date, '%Y%m%d')
        end_date_dt = datetime.strptime(end_date, '%Y%m%d')
    except ValueError:
        raise ValueError("Invalid date format. Please use YYYYMMDD.")
    
    async with conn.cursor() as cur:
        await cur.execute("""
        SELECT id, text, scores, sentiment, cleaned_date
        FROM entries
        WHERE cleaned_date BETWEEN ? AND ?
        """, (start_date, end_date))
        rows = await cur.fetchall()
        # Convert scores back from JSON string to a dictionary
        rows = [(row[0], row[1], json.loads(row[2]), row[3], row[4]) for row in rows]
        sorted_rows = sorted(rows, key=lambda row: row[4])  # Sort the rows by the 5th element (cleaned_date)
        return sorted_rows

async def search_tweets_by_keyword(conn, keyword: str) -> list:
    """
    Search tweets in the SQLite database by a keyword.

    Args:
        conn (aiosqlite.Connection): The database connection.
        keyword (str): The keyword to search for in the tweets.

    Returns:
        list: A list of tuples representing tweets that contain the keyword.
    """
    async with conn.cursor() as cur:
        await cur.execute("""
        SELECT id, text
        FROM entries
        WHERE cleaned_text LIKE ?
        """, ('%' + keyword + '%',))
        rows = await cur.fetchall()
        return rows
    
async def get_word_counts_db(conn):
    """
    Create a dictionary containing each word and how many times it is used in the database.

    Args:
        conn (aiosqlite.Connection): The database connection.

    Returns:
        dict: A dictionary with words as keys and their counts as values, sorted from highest to lowest count.
    """
    async with conn.cursor() as cur:
        await cur.execute("SELECT cleaned_text FROM entries")
        rows = await cur.fetchall()

        # Flatten the list of tuples and join all texts into one large string
        all_texts = ' '.join(row[0] for row in rows)
        
        # Tokenize and count words
        words = re.findall(r'\b\w+\b', all_texts.lower())
        word_counts = Counter(words)
        
        # Sort words by count in descending order
        sorted_word_counts = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True))
        
        return sorted_word_counts