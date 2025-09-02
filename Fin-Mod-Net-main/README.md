# FinModNet: News Aware Modulated LSTM Framework

In this project, we explored the integration of financial data and news sentiment to enhance the prediction of stock prices utilising modulated LSTM and Word2Vec. 

Steps and instructions for execution:

1. Ensure that the following files are in the same directory:
    (i) combined_stock_data_jan_apr_2025.csv
    (ii) Filtered_Headlines_Jan-April_2025.csv
    (iii) combined_stock_data_oct_dec_2024.csv
    (iv) Filtered_Headlines_Oct-Dec_2024.csv
    (v) mod_lstm_final.py

    NOTE: REPLACE THE FILE PATHS IN THE SOURCE CODE WITH THE RELATIVE PATH FOR YOUR FILES IN YOUR WORKING DIRECTORY

2. Following dependencies are accounted for:
    (i) pandas
    (ii) numpy
    (iii) torch (i.e PyTorch)
    (iv) gensim (for Word2Vec)
    (v) sklearn
    (vi) tqdm

3. Once Steps (1) and (2) are observed, and you are in the correponding working directory, then simply execute the file mod_lstm_final.py in your terminal with the command "python mod_lstm_final.py"