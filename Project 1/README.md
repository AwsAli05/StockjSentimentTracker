# Financial News Summarizer

**Created By Aws Ali, 2024-05-05**

The Financial News Summarizer is a tool designed to fetch, summarize, and analyze financial news articles for selected stock tickers. It uses a pre-trained machine learning model to generate concise summaries and perform sentiment analysis on news articles from Google News.

## Features

- **News Fetching**: Automatically scrapes news articles from Google News based on specified stock tickers.
- **Summarization**: Uses the Pegasus summarization model to create brief summaries of each article.
- **Sentiment Analysis**: Applies sentiment analysis to each summary to gauge the overall sentiment of the news.
- **Configurable**: Users can specify which tickers to fetch news for through a simple configuration file.

## Requirements

- Windows, or Linux
- An internet connection for fetching news articles
- CUDA 11.8 or later https://developer.nvidia.com/cuda-11-8-0-download-archive?target_os=Windows
- Pytorch 2.2 or later
- BeautifulSoup4
- Transformers
- Cloudscrapers

## Setup and Installation

1. **Download the Executable**:
   - Download `FinancialNewsSummarizer.exe` from the provided link. **DONT FORGET TO PUT IN A GITHUB LINK**

2. **Configuration File**:
   - Create a file named `monitored_tickers.txt` in the same directory as the executable.
   - List the tickers you want to monitor, separated by commas. Example format:
     ```
     TSLA, AAPL, GOOG, MSFT
     ```
3. **Requirements File**
Before running the Financial News Summarizer, you need to install the necessary Python libraries. Open a terminal or command prompt and navigate to the directory containing the `requirements.txt` file. Run the following command:

```bash
pip install -r requirements.txt
```
## Usage

To run the Financial News Summarizer, simply execute the `FinancialNewsSummarizer.exe` file. Ensure that the `monitored_tickers.txt` file is in the same directory as the executable and is properly formatted.

The program will read the tickers from the `monitored_tickers.txt` file, fetch the relevant news articles, and then display the summaries and sentiment analysis results in the console. It will also generate a CSV file, `assetsummaries.csv`, containing the summaries, sentiment labels, confidence scores, and URLs of the articles.

## Customization

- **Modifying Tickers**: Change the tickers in the `monitored_tickers.txt` file as needed. There is no limit to the number of tickers you can add; however, more tickers might increase the runtime of the tool.

## Support

If you encounter any issues or have questions, please feel free to contact the support team at [aliaws123@outlook.com].

## License

This software is provided under the [MIT License](LICENSE).
