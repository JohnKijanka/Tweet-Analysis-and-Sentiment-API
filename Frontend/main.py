import streamlit as st
import requests

# Define the API base URL
BASE_URL = "http://127.0.0.1:8000"

# Streamlit app title
st.title("Tweet Sentiment Simulator")

# Section for Process String
st.header("Process Tweet")

input_string = st.text_input("Enter a Tweet to process:", "")
if st.button("Process String"):
    if input_string:
        response = requests.get(f"{BASE_URL}/process_string", params={"input_string": input_string})
        if response.status_code == 200:
            result = response.json()
            st.write(f"**Text:** {result['text']}")
            st.write(f"**Scores:** {result['score']}")
            st.write(f"**Sentiment:** {result['sentiment']}")
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
    else:
        st.warning("Please enter a Tweet to process.")

# Section for Read Random Entry
st.header("Read Random Entry")

if st.button("Read Random Entry"):
    response = requests.get(f"{BASE_URL}/entries/random")
    if response.status_code == 200:
        result = response.json()
        st.write(f"**Text:** {result['text']}")
        st.write(f"**Scores:** {result['scores']}")
        st.write(f"**Sentiment:** {result['sentiment']}")
    else:
        st.error(f"Error: {response.status_code} - {response.text}")

# Section for Date filter
st.header("Filter Entries by Date")

# Input widgets
start_date = st.date_input("Start Date")
end_date = st.date_input("End Date")

if st.button("Filter Entries"):
    if start_date and end_date:
        # Check if start_date is before end_date
        if start_date > end_date:
            st.write("Error: Start Date cannot be after End Date.")
        else:
            # Convert dates to YYYYMMDD format
            start_date_str = start_date.strftime('%Y%m%d')
            end_date_str = end_date.strftime('%Y%m%d')

            # Call FastAPI endpoint
            response = requests.get(f"{BASE_URL}/filter_dates/", params={"start_date": start_date_str, "end_date": end_date_str})
            
            if response.status_code == 200:
                entries = response.json()
                if entries:
                    for entry in entries:
                        st.write(f"**Text:** {entry[1]}")
                        st.write(f"**Date:** {entry[4]}")
                        st.write("---")  # Add a separator between entries
                else:
                    st.write("No entries found for the selected date range.")
            else:
                st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("Please select both start and end dates.")

st.header("Search Tweets by Keyword")

# Input widget
keyword = st.text_input("Keyword")

if st.button("Search Tweets"):
    if keyword:
        # Call FastAPI endpoint
        response = requests.get(f"{BASE_URL}/search_tweets/", params={"keyword": keyword})
        
        if response.status_code == 200:
            results = response.json()
            if results:
                for tweet in results:
                    st.write(f"**Text:** {tweet[1]}")
                    st.write("---")  # Add a separator between entries
            else:
                st.write("No tweets found for the given keyword.")
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("Please enter a keyword.")

st.header("Top N Words in Tweets")

# Input widget for number of top words
n = st.number_input("Number of top words to display", min_value=1, value=10, step=1)

if st.button("Get Top Words"):
    if n > 0:
        # Call FastAPI endpoint with the number of top words to display
        response = requests.get(f"{BASE_URL}/word_counts/")
        
        if response.status_code == 200:
            word_counts = response.json()
            if word_counts:
                # Sort words by count in descending order and get the top N
                top_words = dict(sorted(word_counts.items(), key=lambda item: item[1], reverse=True)[:n])
                
                st.write(f"Top {n} words sorted from highest to lowest:")
                for word, count in top_words.items():
                    st.write(f"**{word}**: {count}")
                    st.write("---")  # Add a separator between entries
            else:
                st.write("No words found.")
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("Please enter a positive number.")

# Streamlit UI
st.header("Fetch Entries by ID Range")

# Input widgets for start and end IDs
start_id = st.number_input("Start ID", min_value=1, value=1, step=1)
end_id = st.number_input("End ID", min_value=1, value=start_id, step=1)

if st.button("Fetch Entries"):
    # If end_id is not provided or is less than start_id, default it to start_id
    if end_id < start_id:
        end_id = start_id

    if start_id <= end_id:
        # Call FastAPI endpoint with the ID range
        response = requests.get(f"{BASE_URL}/entries_by_id/", params={"start_id": start_id, "end_id": end_id})
        
        if response.status_code == 200:
            entries = response.json()
            if entries:
                st.write(f"Entries from ID {start_id} to {end_id}:")
                # Display the results directly without converting to a DataFrame
                for entry in entries:
                    st.write(f"**ID:** {entry['id']}")
                    st.write(f"**Text:** {entry['text']}")
                    st.write(f"**Sentiment:** {entry['sentiment']}")
                    st.write("---")  # Add a separator between entries
            else:
                st.write("No entries found for the selected ID range.")
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("End ID must be greater than or equal to Start ID.")

st.header("Find Top N Similar Texts")

# Input widgets for ID and top N
id_input = st.number_input("Enter the ID:", min_value=1, step=1)
top_n_input = st.number_input("Enter the number of similar texts to retrieve (N):", min_value=1, step=1)

if st.button("Find Similar Texts"):
    # Ensure the inputs are valid
    if id_input > 0 and top_n_input > 0:
        # Call the FastAPI endpoint
        response = requests.get(f"{BASE_URL}/top_n_similar_texts/", params={"id": id_input, "top_n": top_n_input})
        original = requests.get(f"{BASE_URL}/entries_by_id/", params={"start_id": id_input, "end_id": id_input})

        if original.status_code == 200:
            entries = original.json()
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            similar_texts = response.json()
            if similar_texts:
                st.write(f"Top {top_n_input} similar texts to {entries[0]['text']}:")
                for text in similar_texts:
                    st.write(f"- {text}")
            else:
                st.write("No similar texts found.")
        else:
            st.write(f"Error: {response.status_code} - {response.text}")
    else:
        st.write("Please enter valid ID and N values.")