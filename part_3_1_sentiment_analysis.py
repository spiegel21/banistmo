import pandas as pd
import matplotlib.pyplot as plt
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import time
from bs4 import BeautifulSoup
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time
import random

# Download necessary NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')

# Part 3.1.a: Download news/comments about EURUSD
class ForexNewsScraper:
    def __init__(self):
        """Initialize the Forex news scraper"""
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
        ]
    
    def get_random_headers(self):
        """Generate random headers to avoid being blocked"""
        return {
            'User-Agent': random.choice(self.user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0'
        }
    

    def scrape_fxstreet_news(self):
        """
        Scrape EUR/USD headlines using Selenium to render JS content.
        """
        all_news = []
        url = "https://www.fxstreet.com/news/latest/asset?dFR[Category][0]=News&dFR[Tags][0]=EURUSD"

        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--disable-gpu')
            options.add_argument('--no-sandbox')
            driver = webdriver.Chrome(options=options)

            driver.get(url)
            time.sleep(5)  # wait for JavaScript to load

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            driver.quit()

            headlines = soup.find_all("h4", class_="fxs_headline_tiny")

            for h in headlines:
                a_tag = h.find("a")
                if a_tag:
                    title = a_tag.text.strip()
                    link = a_tag['href']
                    all_news.append({"title": title, "url": link})

        except Exception as e:
            print(f"[Scraping Error] {e}")

        return all_news

    
    def get_forex_news(self, use_sample=False):
        """
        Get Forex news from various sources or use sample data
        
        Parameters:
        use_sample (bool): Whether to use sample data instead of web scraping
        
        Returns:
        list: List of dictionaries with news data
        """
        if use_sample:
            print("Using sample news data...")
            return self.load_sample_data()
        
        all_news = []
        
        # Scrape from multiple sources
        print("Scraping news from FXStreet...")
        investing_news = self.scrape_fxstreet_news()
        all_news.extend(investing_news)
        
        # If no news found, use sample data
        if not all_news:
            print("No news found. Using sample data...")
            return self.load_sample_data()
        
        print(f"Total news articles: {len(all_news)}")
        print(all_news)
        return all_news

# Part 3.1.b: Sentiment Analysis
class ForexSentimentAnalyzer:
    def __init__(self, analyzer_type='vader'):
        """
        Initialize the sentiment analyzer
        
        Parameters:
        analyzer_type (str): Type of analyzer to use ('vader' or 'textblob')
        """
        self.analyzer_type = analyzer_type
        
        if analyzer_type == 'vader':
            self.analyzer = SentimentIntensityAnalyzer()
        
        # Forex specific terms that might indicate bullish or bearish sentiment
        self.forex_bullish_terms = [
            'bullish', 'rise', 'rising', 'strengthen', 'uptrend', 'higher', 'rebound',
            'recover', 'growth', 'gain', 'positive', 'upside', 'rally', 'surge', 'advance',
            'strong euro', 'euro strength', 'dollar weakness', 'weak dollar'
        ]
        
        self.forex_bearish_terms = [
            'bearish', 'fall', 'falling', 'weaken', 'downtrend', 'lower', 'decline',
            'drop', 'decrease', 'negative', 'downside', 'retreat', 'plunge', 'tumble',
            'weak euro', 'euro weakness', 'dollar strength', 'strong dollar'
        ]
    
    def analyze_with_vader(self, text):
        """
        Analyze sentiment using VADER
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        float: Compound sentiment score
        """
        scores = self.analyzer.polarity_scores(text)
        return scores['compound']
    
    def analyze_with_textblob(self, text):
        """
        Analyze sentiment using TextBlob
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        float: Polarity score
        """
        blob = TextBlob(text)
        return blob.sentiment.polarity
    
    def analyze_forex_specific_terms(self, text):
        """
        Analyze text for Forex-specific bullish/bearish terms
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        float: Forex-specific sentiment score (positive=bullish, negative=bearish)
        """
        text_lower = text.lower()
        
        # Count bullish and bearish terms
        bullish_count = sum(term in text_lower for term in self.forex_bullish_terms)
        bearish_count = sum(term in text_lower for term in self.forex_bearish_terms)
        
        # Calculate score (-1 to 1 range)
        if bullish_count == 0 and bearish_count == 0:
            return 0
        else:
            return (bullish_count - bearish_count) / (bullish_count + bearish_count)
    
    def analyze_sentiment(self, text):
        """
        Analyze sentiment of text using multiple methods
        
        Parameters:
        text (str): Text to analyze
        
        Returns:
        dict: Dictionary with sentiment scores
        """
        # Use selected analyzer
        general_score = self.analyze_with_vader(text) if self.analyzer_type == 'vader' else self.analyze_with_textblob(text)
        
        # Get forex-specific sentiment
        forex_score = self.analyze_forex_specific_terms(text)
        
        # Get final score (weighted average, with more weight on forex-specific terms)
        final_score = 0.4 * general_score + 0.6 * forex_score if forex_score != 0 else general_score
        
        # Determine sentiment
        if final_score > 0.1:
            sentiment = "bullish"
        elif final_score < -0.1:
            sentiment = "bearish"
        else:
            sentiment = "neutral"
        
        return {
            'general_score': general_score,
            'forex_score': forex_score,
            'final_score': final_score,
            'sentiment': sentiment
        }

# Part 3.1.c: Automatic sentiment analysis process
class ForexSentimentAnalysisSystem:
    def __init__(self, analyzer_type='vader'):
        """
        Initialize the forex sentiment analysis system
        
        Parameters:
        analyzer_type (str): Type of analyzer to use ('vader' or 'textblob')
        """
        self.scraper = ForexNewsScraper()
        self.analyzer = ForexSentimentAnalyzer(analyzer_type)
        self.results_df = None
    
    def analyze_news_data(self, news_data):
        """
        Analyze sentiment for a list of news articles
        
        Parameters:
        news_data (list): List of dictionaries with news data
        
        Returns:
        pd.DataFrame: DataFrame with sentiment analysis results
        """
        results = []
        
        for article in news_data:
            # Create text to analyze (title + summary)
            text = article['title']
            
            # Analyze sentiment
            sentiment_results = self.analyzer.analyze_sentiment(text)
            
            # Add to results
            results.append({
                'title': article['title'],
                'url': article['url'],
                'sentiment': sentiment_results['sentiment'],
                'sentiment_score': sentiment_results['final_score']
            })
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save the results
        self.results_df = df
        
        return df
    
    def save_results_to_csv(self, filename='out/forex_sentiment_results.csv'):
        """
        Save sentiment analysis results to CSV
        
        Parameters:
        filename (str): Name of the CSV file
        """
        if self.results_df is not None:
            self.results_df.to_csv(filename, index=False)
            print(f"Results saved to {filename}")
        else:
            print("No results to save")
    
    def generate_sentiment_summary(self):
        """
        Generate a summary of sentiment analysis results
        
        Returns:
        dict: Dictionary with sentiment summary
        """
        if self.results_df is None or len(self.results_df) == 0:
            return {"error": "No sentiment data available"}
        
        # Count sentiments
        sentiment_counts = self.results_df['sentiment'].value_counts()
        bullish_count = sentiment_counts.get('bullish', 0)
        bearish_count = sentiment_counts.get('bearish', 0)
        neutral_count = sentiment_counts.get('neutral', 0)
        
        # Calculate average sentiment score
        avg_sentiment_score = self.results_df['sentiment_score'].mean()
        
        # Determine overall sentiment
        if bullish_count > bearish_count and bullish_count > neutral_count:
            overall_sentiment = "bullish"
        elif bearish_count > bullish_count and bearish_count > neutral_count:
            overall_sentiment = "bearish"
        else:
            if avg_sentiment_score > 0.05:
                overall_sentiment = "slightly bullish"
            elif avg_sentiment_score < -0.05:
                overall_sentiment = "slightly bearish"
            else:
                overall_sentiment = "neutral"
        
        # Find most positive and negative articles
        most_positive_idx = self.results_df['sentiment_score'].idxmax()
        most_negative_idx = self.results_df['sentiment_score'].idxmin()
        
        most_positive_article = self.results_df.iloc[most_positive_idx]
        most_negative_article = self.results_df.iloc[most_negative_idx]
        
        # Create summary
        summary = {
            'overall_sentiment': overall_sentiment,
            'average_sentiment_score': avg_sentiment_score,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': neutral_count,
            'total_articles': len(self.results_df),
            'most_positive_article': {
                'title': most_positive_article['title'],
                'score': most_positive_article['sentiment_score']
            },
            'most_negative_article': {
                'title': most_negative_article['title'],
                'score': most_negative_article['sentiment_score']
            }
        }
        
        return summary
    
    def plot_sentiment_distribution(self):
        """
        Plot the distribution of sentiment results
        """
        if self.results_df is None or len(self.results_df) == 0:
            print("No sentiment data available for plotting")
            return
        
        # Create figure
        plt.figure(figsize=(12, 10))
        
        # Plot 1: Sentiment counts
        plt.subplot(2, 2, 1)
        counts = self.results_df['sentiment'].value_counts()
        colors = {'bullish': 'green', 'neutral': 'gray', 'bearish': 'red'}
        counts.plot(kind='bar', color=[colors.get(x, 'blue') for x in counts.index])
        plt.title('Distribution of Sentiments')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        
        # Plot 2: Sentiment scores histogram
        plt.subplot(2, 2, 2)
        plt.hist(self.results_df['sentiment_score'], bins=15, color='skyblue', edgecolor='black')
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title('Distribution of Sentiment Scores')
        plt.xlabel('Sentiment Score')
        plt.ylabel('Count')
        
        plt.tight_layout()
        plt.show()
    
    def run_analysis(self, use_sample_data=True):
        """
        Run the complete sentiment analysis process
        
        Parameters:
        use_sample_data (bool): Whether to use sample data instead of web scraping
        
        Returns:
        tuple: (DataFrame with results, sentiment summary)
        """
        # Step 1: Get news data
        news_data = self.scraper.get_forex_news(use_sample=use_sample_data)
        
        # Step 2: Analyze sentiment
        results_df = self.analyze_news_data(news_data)
        
        # Step 3: Generate summary
        summary = self.generate_sentiment_summary()
        
        # Step 4: Display results
        print("\n===== FOREX SENTIMENT ANALYSIS RESULTS =====")
        print(f"Total articles analyzed: {len(results_df)}")
        print(f"Overall market sentiment: {summary['overall_sentiment']}")
        print(f"Average sentiment score: {summary['average_sentiment_score']:.4f}")
        print(f"Bullish articles: {summary['bullish_count']} ({summary['bullish_count']/len(results_df)*100:.1f}%)")
        print(f"Bearish articles: {summary['bearish_count']} ({summary['bearish_count']/len(results_df)*100:.1f}%)")
        print(f"Neutral articles: {summary['neutral_count']} ({summary['neutral_count']/len(results_df)*100:.1f}%)")
        
        print("\nMost positive article:")
        print(f"- {summary['most_positive_article']['title']}")
        print(f"- Score: {summary['most_positive_article']['score']:.4f}")
        
        print("\nMost negative article:")
        print(f"- {summary['most_negative_article']['title']}")
        print(f"- Score: {summary['most_negative_article']['score']:.4f}")
        
        # Plot results
        self.plot_sentiment_distribution()
        
        # Save results
        self.save_results_to_csv()
        
        return results_df, summary

# Part 3.1.d: Trading Implications Analysis
def analyze_trading_implications(sentiment_summary):
    """
    Analyze the trading implications of the sentiment analysis
    
    Parameters:
    sentiment_summary (dict): Dictionary with sentiment summary
    
    Returns:
    str: Analysis of trading implications
    """
    overall_sentiment = sentiment_summary['overall_sentiment']
    avg_score = sentiment_summary['average_sentiment_score']
    bullish_pct = sentiment_summary['bullish_count'] / sentiment_summary['total_articles'] * 100
    bearish_pct = sentiment_summary['bearish_count'] / sentiment_summary['total_articles'] * 100
    
    analysis = "=== TRADING IMPLICATIONS ANALYSIS ===\n\n"
    
    # Overall sentiment analysis
    analysis += f"The overall market sentiment for EURUSD is {overall_sentiment.upper()} "
    analysis += f"with an average sentiment score of {avg_score:.4f}.\n\n"
    
    # Strength of sentiment
    if abs(avg_score) > 0.3:
        sentiment_strength = "strong"
    elif abs(avg_score) > 0.1:
        sentiment_strength = "moderate"
    else:
        sentiment_strength = "weak"
    
    analysis += f"This represents a {sentiment_strength} sentiment signal.\n\n"
    
    # Trading strategy implications
    analysis += "Trading Strategy Implications:\n"
    
    if overall_sentiment in ['bullish', 'slightly bullish']:
        analysis += "1. Bullish sentiment suggests potential upward pressure on EURUSD.\n"
        if bullish_pct > 60:
            analysis += "2. Strong bullish consensus (>{:.1f}%) could indicate a trend continuation if price is already rising.\n".format(bullish_pct)
            analysis += "3. Risk: If extremely bullish (>{:.1f}%), consider potential for contrarian moves as the market may be overcrowded.\n".format(bullish_pct)
        
        analysis += "4. Technical considerations:\n"
        analysis += "   - Look for bullish confirmation from price action and technical indicators.\n"
        analysis += "   - Consider long positions on pullbacks to support levels.\n"
        analysis += "   - Move average crossover strategy: Confirm with bullish signals from MA50/MA200 before entering.\n\n"
    
    elif overall_sentiment in ['bearish', 'slightly bearish']:
        analysis += "1. Bearish sentiment suggests potential downward pressure on EURUSD.\n"
        if bearish_pct > 60:
            analysis += "2. Strong bearish consensus (>{:.1f}%) could indicate a trend continuation if price is already falling.\n".format(bearish_pct)
            analysis += "3. Risk: If extremely bearish (>{:.1f}%), consider potential for relief rallies as the market may be oversold.\n".format(bearish_pct)
        
        analysis += "4. Technical considerations:\n"
        analysis += "   - Look for bearish confirmation from price action and technical indicators.\n"
        analysis += "   - Consider short positions on rallies to resistance levels.\n"
        analysis += "   - Move average crossover strategy: Confirm with bearish signals from MA50/MA200 before entering.\n\n"
    
    else:  # neutral
        analysis += "1. Neutral sentiment suggests a potential range-bound market for EURUSD.\n"
        analysis += "2. Consider range-trading strategies rather than trend-following approaches.\n"
        analysis += "3. Technical considerations:\n"
        analysis += "   - Pay closer attention to support and resistance levels.\n"
        analysis += "   - Moving average crossover strategy: May generate false signals in rangebound conditions.\n"
        analysis += "   - Consider using oscillators like RSI to identify overbought/oversold conditions.\n\n"
    
    # Risk management reminders
    analysis += "Risk Management Recommendations:\n"
    analysis += "1. Use sentiment as a complementary indicator, not as the sole basis for trading decisions.\n"
    analysis += "2. Combine sentiment analysis with technical and fundamental analysis for more reliable signals.\n"
    analysis += "3. News sentiment can shift rapidly - maintain appropriate position sizing and stop-loss orders.\n"
    analysis += "4. Consider reduced position sizes during periods of mixed or weak sentiment signals.\n\n"
    
    # Integration with MA strategy
    analysis += "Integration with Moving Average Crossover Strategy:\n"
    
    if overall_sentiment in ['bullish', 'slightly bullish']:
        analysis += "- Strongest setup: Bullish sentiment + MA50 crossing above MA200\n"
        analysis += "- Consider extending profit targets on long positions\n"
        analysis += "- Be more patient with bullish positions even during minor pullbacks\n"
    elif overall_sentiment in ['bearish', 'slightly bearish']:
        analysis += "- Strongest setup: Bearish sentiment + MA50 crossing below MA200\n"
        analysis += "- Consider extending profit targets on short positions\n"
        analysis += "- Be more patient with bearish positions even during minor rallies\n"
    else:
        analysis += "- Neutral sentiment may reduce effectiveness of the MA crossover strategy\n"
        analysis += "- Consider tighter profit targets in both directions\n"
        analysis += "- Pay closer attention to support/resistance for entry/exit points\n"
    
    return analysis

# Part 3.1.e: Main execution function
def main():
    """
    Main function to run the entire sentiment analysis system
    """
    print("=== FOREX SENTIMENT ANALYSIS SYSTEM ===")
    print("This system analyzes news sentiment for EURUSD and provides trading implications.")
    
    # Initialize the system
    system = ForexSentimentAnalysisSystem(analyzer_type='vader')
    
    # Run the analysis
    # Note: Set use_sample_data to False to use real-time web scraping (requires internet connection)
    results_df, summary = system.run_analysis(use_sample_data=False)
    
    # Analyze trading implications
    implications = analyze_trading_implications(summary)
    
    # Print trading implications
    print("\n" + implications)
    
    # Save implications to file
    with open('out/trading_implications.txt', 'w') as f:
        f.write(implications)
    
    print("\nAnalysis complete. Results saved to 'forex_sentiment_results.csv' and 'trading_implications.txt'")
    
    return results_df, summary, implications

# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()