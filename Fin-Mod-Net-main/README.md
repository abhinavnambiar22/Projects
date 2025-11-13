# FinModNet: A News-Aware Modulated LSTM Framework for Stock Forecasting

### 概要

This project implements **FinModNet**, a novel framework designed to improve stock price prediction by integrating both historical financial data and qualitative news sentiment.

The core innovation is a **modulated LSTM network** where the model's weights are dynamically generated based on the semantic content of financial news headlines. This allows the model to adapt its predictions in real-time, accounting for market sentiment, which traditional time-series models often miss.

### Core Concept: How It Works

1.  **Data Integration**: The model takes two inputs for each 7-day window:
    * **Stock Data**: Normalized 7-day sequences of OHLCV (Open, High, Low, Close, Volume) data.
    * **News Data**: All financial news headlines from the same 7-day period.

2.  **News Embedding**: News headlines are tokenized and processed using a **Word2Vec (Skip-gram) model** trained on the financial news corpus. This converts text into 300-dimensional vectors that capture semantic meaning. These vectors are then averaged to create a single "daily news vector".

3.  **Dynamic Weight Modulation**: This is the key. Instead of a static LSTM, the averaged 7-day news vector is fed into a separate "modulation network" (inspired by HyperNetworks). This network *dynamically generates* the `W_ih`, `W_hh`, `b_ih`, and `b_hh` weights and biases for the main LSTM for that specific time step.

4.  **Prediction**: The LSTM, now "aware" of the news sentiment via its custom-generated weights, processes the 7-day stock data sequence to predict the closing price for the *next day*.

### Performance 📈

The model was trained on data from October-December 2024 and evaluated on unseen data from January-April 2025.

* **Overall MAPE:** 3.21%
* **Overall MAE:** $2.68
* **Overall RMSE:** $3.45

The results showed that incorporating this news-driven modulation **significantly improved predictive accuracy** over baseline models that rely only on stock history. For a detailed breakdown of performance by company (Apple, Microsoft, Tesla, etc.) and full methodology, please see the [DL_Final_report.pdf](DL_Final_report.pdf).

### Tech Stack 🛠️

* **PyTorch**
* **Gensim** (for Word2Vec)
* **Pandas**
* **NumPy**
* **Scikit-learn** (for `MinMaxScaler`)

### How to Run 🚀

1.  **Clone the Repository**
    ```bash
    git clone [YOUR_REPOSITORY_LINK]
    cd FinMod-Net-main
    ```

2.  **Install Dependencies**
    Ensure you have all the required libraries installed:
    ```bash
    pip install pandas numpy torch gensim scikit-learn tqdm
    ```

3.  **Data Files**
    Make sure the following 5 files are present in your working directory:
    * `combined_stock_data_jan_apr_2025.csv` (for testing)
    * `Filtered_Headlines_Jan-April_2025.csv` (for testing)
    * `combined_stock_data_oct_dec_2024.csv` (for training)
    * `Filtered_Headlines_Oct-Dec_2024.csv` (for training)
    * `mod_lstm_final.py`

4.  **IMPORTANT: Update File Paths**
    You **MUST** update the hardcoded file paths inside `mod_lstm_final.py` to match the relative paths in your local directory.

    *Look for these lines in the script and change them:*
    ```python
    # Change these paths!
    stock_df = load_stock_data(r"combined_stock_data_oct_dec_2024.csv")
    news_df = load_headlines(r"Filtered_Headlines_Oct-Dec_2024.csv")
    
    # ...and in the testing section...
    test_stock_df = pd.read_csv(r'combined_stock_data_jan_apr_2025.csv')
    test_news_df = pd.read_csv(r"Filtered_Headlines_Jan-April_2025.csv")
    ```

5.  **Execute the Script**
    Once the paths are corrected, run the script from your terminal:
    ```bash
    python mod_lstm_final.py
    ```
    The script will first train the Word2Vec model, then train the `LSTMHyperModel`, and finally, it will automatically run the evaluation on the test set, printing the performance metrics and predictions.
