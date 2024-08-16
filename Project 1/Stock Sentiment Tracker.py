import concurrent.futures
import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline
from bs4 import BeautifulSoup
import cloudscraper
import re
import time
import csv

# Load the Pegasus model for summarization and move it to GPU for faster processing
model_name = 'human-centered-summarization/financial-summarization-pegasus'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
model.to('cuda')  # Move the model to GPU

MAX_LENGTH = 512  # Define the maximum sequence length for Pegasus

# Define the list of tickers to monitor and a list of URL parts to exclude
def get_tickers_from_file(filepath='monitored_tickers.txt'):
    with open(filepath, 'r') as file:
        tickers = file.read().strip().split(',')
        tickers = [ticker.strip() for ticker in tickers]  # Clean up any extra whitespace
    return tickers

monitored_tickers = get_tickers_from_file()
exclude_list = ['maps', 'accounts', 'support', 'policies', 'preferences'] # Makes sure to not grab google support urls etc.

# Load a sentiment analysis pipeline for analyzing financial news
sentiment = pipeline("text-classification", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis")


def fetch_urls(ticker):
    """Fetch URLs from Google News search for a given ticker."""
    search_url = f'https://www.google.com/search?q=yahoo+finance+{ticker}&tbm=nws'
    scraper = cloudscraper.create_scraper()
    r = scraper.get(search_url)
    soup = BeautifulSoup(r.text, 'html.parser')
    atags = soup.find_all('a')
    return [link['href'] for link in atags if link.has_attr('href')]

def strip_unwanted_urls(urls):
    """Filter out unwanted URLs based on the exclude list."""
    return list(set([re.findall(r'(https?://\S+)', url)[0].split('&')[0] for url in urls if 'https://' in url and not any(exclude_word in url for exclude_word in exclude_list)]))

def fetch_article(url):
    """Fetch and extract text from the article at the given URL."""
    scraper = cloudscraper.create_scraper()
    r = safe_get_request(url, scraper)
    if r is None:  # If the response is None, return an empty string or None
        return None
    soup = BeautifulSoup(r.text, 'html.parser')
    paragraphs = soup.find_all('p')
    return ' '.join(paragraph.text for paragraph in paragraphs)

def safe_get_request(url, scraper, max_retries=3):
    """Attempt to get a response from the URL with retries on failure."""
    for attempt in range(max_retries):
        try:
            response = scraper.get(url)
            response.raise_for_status()
            return response
        except Exception as e:
            time.sleep(2 ** attempt)  # Exponential backoff
    print(f"Failed to fetch {url} after {max_retries} attempts")
    return None

def summarize(articles):
    """Generate summaries for a list of articles using the Pegasus model."""
    # Filter out None values in articles
    valid_articles = [article for article in articles if article is not None]
    if not valid_articles:  # If all articles are None, skip processing
        return []
    inputs = tokenizer(valid_articles, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to('cuda')
    outputs = model.generate(**inputs, max_length=55, num_beams=5, early_stopping=True)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)

def create_output_array(summaries, scores, urls):
    """Create a structured output array for saving to a CSV file."""
    output = []
    for ticker in monitored_tickers:
        for counter in range(len(summaries[ticker])):
            output.append([
                ticker,
                summaries[ticker][counter],
                scores[ticker][counter]['label'],
                scores[ticker][counter]['score'],
                urls[ticker][counter]
            ])
    return output

# Measure the start time of the script to calculate total runtime later
start_time = time.time()

# Use ThreadPoolExecutor to parallelize fetching URLs and articles
try:
    with concurrent.futures.ThreadPoolExecutor() as executor:
        raw_urls = {ticker: executor.submit(fetch_urls, ticker) for ticker in monitored_tickers}
        print('Scraped URLS')
        cleaned_urls = {ticker: strip_unwanted_urls(raw_urls[ticker].result()) for ticker in monitored_tickers}
        print('cleaned URLS')
        articles_futures = {ticker: [executor.submit(fetch_article, url) for url in cleaned_urls[ticker]] for ticker in monitored_tickers}
        articles = {ticker: [future.result() for future in articles_futures[ticker]] for ticker in monitored_tickers}
        print('Scraped Paragraphs')
        print('Generating Summaries. (Takes a while please do not exit)')
        summaries = {ticker: summarize(articles[ticker]) for ticker in monitored_tickers}
        print('Generated Summaries')
        print("Calculating Sentiment")
        scores = {ticker: sentiment(summaries[ticker]) for ticker in monitored_tickers}
        print("Calculated Sentiment")

    # Create an output array and save to CSV
    final_output = create_output_array(summaries, scores, cleaned_urls)
    final_output.insert(0, ['Ticker', 'Summary', 'Label', 'Confidence', 'URL'])
    with open('assetsummaries.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerows(final_output)
    print('Exported Data')
    # Print total execution time
    print(f"Total execution time: {time.time() - start_time} seconds")
    input("Press Enter to close the program...")

except Exception as e:
    print(f"An error occurred: {e}")
    input("Press Enter to close the program...")