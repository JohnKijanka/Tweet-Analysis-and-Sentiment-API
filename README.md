# Tweet-Analysis-and-Sentiment-API

## Overview

This project is designed to analyze and manage tweet data efficiently. It includes a backend for database operations and a frontend for user interaction. The main functionalities include sentiment analysis, reading and filtering database entries, and finding similar tweets.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.12.4
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:

    '''bash
    git clone https://github.com/JohnKijanka/Tweet-Analysis-and-Sentiment-API.git
    cd Tweet-Analysis-and-Sentiment-API
    '''

2. Install the required Python packages:

    '''bash
    pip install -r requirements.txt
    '''

### Running the Application

#### Backend

To start the backend, run:

'''bash
python .\Backend\main.py
'''

#### Frontend

To start the frontend, run:

'''bash
streamlit run .\Frontend\main.py
'''

## Usage

### Sentiment Analysis

* Navigate to the sentiment analysis section in the frontend.

* Enter the string you want to analyze.

* View the sentiment results.

### Reading Entries

* Random Entry: Click the button to fetch a random entry.

* All Entries: Click the button to retrieve and view all entries.

* Filter by Date: Input the date range and click to filter entries.

* Search by Keyword: Enter the keyword and click to search.

* Top N Words: Specify the number of top words and click to find them.

* Entries by ID Range: Input the ID range and click to display the entries.

  - Also shows sentiment results for each entry.

* Top N Similar Texts: Enter the entry ID and specify N to find similar texts.