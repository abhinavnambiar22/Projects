import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gensim.models import Word2Vec
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader

# ---------- Step 1: Load & Preprocess Data ----------

def load_stock_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values(by='date', inplace=True)
    return df

def load_headlines(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'])
    return df

stock_df = load_stock_data(r"C:\Users\Pranav\Python_programs\combined_stock_data_oct_dec_2024.csv")
news_df = load_headlines(r"C:\Users\Pranav\Downloads\Filtered_Headlines_Oct-Dec_2024.csv")

# Normalize stock data (per company)
scalers = {}
for company in stock_df['company'].unique():
    scaler = MinMaxScaler()
    mask = stock_df['company'] == company
    stock_df.loc[mask, ['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(
        stock_df.loc[mask, ['open', 'high', 'low', 'close', 'volume']]
    )
    scalers[company] = scaler

# ---------- Step 2: Prepare Word2Vec Model ----------

all_headlines = [row.split() for row in news_df['headline']]
w2v_model = Word2Vec(sentences=all_headlines, vector_size=300, window=5, min_count=1)

def get_daily_news_vector(date):
    headlines = news_df[news_df['date'] == date]['headline']
    vecs = []
    for line in headlines:
        words = line.split()
        word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vecs:
            vecs.append(np.mean(word_vecs, axis=0))
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)

# ---------- Step 3: Create Dataset ----------

class StockNewsDataset(Dataset):
    def _init_(self, stock_df, window_size=7):
        self.samples = []
        companies = stock_df['company'].unique()
        for company in companies:
            df = stock_df[stock_df['company'] == company].reset_index(drop=True)
            for i in range(len(df) - window_size):
                stock_window = df.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
                date_window = df.iloc[i:i+window_size]['date'].values
                news_vectors = [get_daily_news_vector(date) for date in date_window]
                label = df.iloc[i + window_size]['close']
                self.samples.append((stock_window, news_vectors, label))

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        stock, news, label = self.samples[idx]
        return (
            torch.tensor(stock, dtype=torch.float32),
            torch.tensor(np.array(news).flatten(), dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32)
        )

dataset = StockNewsDataset(stock_df)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---------- Step 4: Define Model ----------

class LSTMHyperModel(nn.Module):
    def _init_(self, input_size=5, hidden_size=64, embed_size=300, lstm_input_days=7):
        super(LSTMHyperModel, self)._init_()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_input_days = lstm_input_days
        news_vector_size = lstm_input_days * embed_size

        self.W_ih_net = nn.Sequential(
            nn.Linear(news_vector_size, hidden_size * 4 * input_size),
            nn.Tanh()
        )
        self.W_hh_net = nn.Sequential(
            nn.Linear(news_vector_size, hidden_size * 4 * hidden_size),
            nn.Tanh()
        )
        self.b_ih_net = nn.Linear(news_vector_size, hidden_size * 4)
        self.b_hh_net = nn.Linear(news_vector_size, hidden_size * 4)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, stock_seq, news_vec):
        batch_size = stock_seq.size(0)
        W_ih = self.W_ih_net(news_vec).view(batch_size, 4 * self.hidden_size, self.input_size)
        W_hh = self.W_hh_net(news_vec).view(batch_size, 4 * self.hidden_size, self.hidden_size)
        b_ih = self.b_ih_net(news_vec).view(batch_size, 4 * self.hidden_size)
        b_hh = self.b_hh_net(news_vec).view(batch_size, 4 * self.hidden_size)

        h_t = torch.zeros(batch_size, self.hidden_size, device=stock_seq.device)
        c_t = torch.zeros(batch_size, self.hidden_size, device=stock_seq.device)

        for t in range(self.lstm_input_days):
            x_t = stock_seq[:, t, :].unsqueeze(2)
            gates = (
                torch.bmm(W_ih, x_t).squeeze(2) +
                torch.bmm(W_hh, h_t.unsqueeze(2)).squeeze(2) +
                b_ih + b_hh
            )
            i_t, f_t, g_t, o_t = gates.chunk(4, dim=1)
            i_t = torch.sigmoid(i_t)
            f_t = torch.sigmoid(f_t)
            g_t = torch.tanh(g_t)
            o_t = torch.sigmoid(o_t)
            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

        return self.output_layer(h_t).squeeze(1)

# ---------- Step 5: Training ----------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMHyperModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for stock, news, label in loader:
        stock, news, label = stock.to(device), news.to(device), label.to(device)
        optimizer.zero_grad()
        preds = model(stock, news)
        loss = criterion(preds, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}: Loss = {epoch_loss/len(loader):.4f}")


#--------------------------TESTING---------------------------------------------------------------------
#--------------------------TESTING---------------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

print('--------------------------TESTING---------------------------------------------------------------------')

# --- Load new test data ---
test_stock_df = pd.read_csv(r'C:\Users\Pranav\Python_programs\django_project\combined_stock_data_jan_apr_2025.csv')
test_news_df = pd.read_csv(r"C:\Users\Pranav\Downloads\Filtered_Headlines_Jan-April_2025.csv")

# --- Preprocess like training ---
test_stock_df['date'] = pd.to_datetime(test_stock_df['date'])
test_news_df['date'] = pd.to_datetime(test_news_df['date'])

# Normalize stock features using company-specific scalers (same as training)
for company in test_stock_df['company'].unique():
    if company in scalers:  # Make sure we have a scaler for this company
        scaler = scalers[company]
        mask = test_stock_df['company'] == company
        test_stock_df.loc[mask, ['open', 'high', 'low', 'close', 'volume']] = scaler.transform(
            test_stock_df.loc[mask, ['open', 'high', 'low', 'close', 'volume']]
        )
    else:
        print(f"Warning: No scaler found for {company}. Skipping this company.")
        test_stock_df = test_stock_df[test_stock_df['company'] != company]

# Use the same news embedding function as in training
def get_daily_news_vector(date, news_df=test_news_df):
    headlines = news_df[news_df['date'] == date]['headline']
    vecs = []
    for line in headlines:
        words = line.split()
        word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vecs:
            vecs.append(np.mean(word_vecs, axis=0))
    return np.mean(vecs, axis=0) if vecs else np.zeros(300)

# --- Create consistent test dataset structure ---
class TestStockNewsDataset(Dataset):
    def _init_(self, stock_df, news_df, window_size=7):
        self.samples = []
        self.companies = []
        self.dates = []
        
        companies = stock_df['company'].unique()
        for company in companies:
            df = stock_df[stock_df['company'] == company].sort_values('date').reset_index(drop=True)
            
            # Ensure we have enough consecutive days
            if len(df) <= window_size:
                continue
                
            for i in range(len(df) - window_size):
                stock_window = df.iloc[i:i+window_size][['open', 'high', 'low', 'close', 'volume']].values
                date_window = df.iloc[i:i+window_size]['date'].values
                news_vectors = [get_daily_news_vector(date) for date in date_window]
                target_date = df.iloc[i + window_size]['date']
                target_value = df.iloc[i + window_size]['close']
                
                self.samples.append((
                    torch.tensor(stock_window, dtype=torch.float32),
                    torch.tensor(np.array(news_vectors).flatten(), dtype=torch.float32),
                    torch.tensor(target_value, dtype=torch.float32)
                ))
                self.companies.append(company)
                self.dates.append(target_date)

    def _len_(self):
        return len(self.samples)

    def _getitem_(self, idx):
        return self.samples[idx]

# Create test dataset
test_dataset = TestStockNewsDataset(test_stock_df, test_news_df)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Evaluation ---
model.eval()
company_preds = {}
company_targets = {}
company_dates = {}

print(f"Evaluating {len(test_dataset)} test samples...")

with torch.no_grad():
    for i in tqdm(range(len(test_dataset))):
        x, news, target = test_dataset[i]
        x = x.unsqueeze(0).to(device)  # Add batch dimension
        news = news.unsqueeze(0).to(device)
        
        # Get prediction
        pred = model(x, news).squeeze().item()
        
        # Store results by company
        company = test_dataset.companies[i]
        date = test_dataset.dates[i]
        
        if company not in company_preds:
            company_preds[company] = []
            company_targets[company] = []
            company_dates[company] = []
            
        company_preds[company].append(pred)
        company_targets[company].append(target.item())
        company_dates[company].append(date)

# --- Calculate and print metrics by company ---
print("\nCompany-wise Evaluation:")
all_preds = []
all_targets = []

for company in company_preds:
    preds = np.array(company_preds[company])
    targets = np.array(company_targets[company])
    
    # Inverse transform the predictions and targets using company-specific scaler
    scaler = scalers[company]
    
    # Create dummy arrays for inverse transform (since we only have 'close')
    dummy_preds = np.zeros((len(preds), 5))
    dummy_targets = np.zeros((len(targets), 5))
    
    # Close price is at index 3 (assuming order is: open, high, low, close, volume)
    dummy_preds[:, 3] = preds
    dummy_targets[:, 3] = targets
    
    # Inverse transform to get actual prices
    inv_preds = scaler.inverse_transform(dummy_preds)[:, 3]
    inv_targets = scaler.inverse_transform(dummy_targets)[:, 3]
    
    all_preds.extend(inv_preds)
    all_targets.extend(inv_targets)
    
    # Calculate metrics
    mae = mean_absolute_error(inv_targets, inv_preds)
    rmse = np.sqrt(mean_squared_error(inv_targets, inv_preds))  # Use np.sqrt with MSE instead of root_mean_squared_error
    
    # Calculate percentage error
    mape = np.mean(np.abs((inv_targets - inv_preds) / inv_targets)) * 100
    
    print(f"{company}: MAE = ${mae:.2f}, RMSE = ${rmse:.2f}, MAPE = {mape:.2f}%")
    
    # Option: Print a few sample predictions vs actual
    print(f"  Sample predictions (last 3 days):")
    for i in range(min(3, len(inv_preds))):
        idx = -(i+1)  # Get last few entries
        date = company_dates[company][idx]
        print(f"  {date.strftime('%Y-%m-%d')}: Predicted=${inv_preds[idx]:.2f}, Actual=${inv_targets[idx]:.2f}")
    print()

# Overall metrics
overall_mae = mean_absolute_error(all_targets, all_preds)
overall_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
overall_mape = np.mean(np.abs((np.array(all_targets) - np.array(all_preds)) / np.array(all_targets))) * 100

print(f"\nOverall: MAE = ${overall_mae:.2f}, RMSE = ${overall_rmse:.2f}, MAPE = {overall_mape:.2f}%")

print("\n--------------------------PREDICTION FOR APRIL 4, 2025--------------------------")

# Make predictions for April 4, 2025
pred_date = pd.Timestamp('2025-04-04')
window_end_date = pred_date - pd.Timedelta(days=1)  # April 3, 2025
window_start_date = window_end_date - pd.Timedelta(days=8)  # 7 days before April 3

for company in test_stock_df['company'].unique():
    if company not in scalers:
        continue
        
    # Get the last 7 days of data before April 4
    comp_data = test_stock_df[test_stock_df['company'] == company].sort_values('date')
    window_data = comp_data[(comp_data['date'] >= window_start_date) & (comp_data['date'] <= window_end_date)]
    
    if len(window_data) != 7:
        print(f"{company}: Insufficient data for prediction window (need 7 days, got {len(window_data)}). Skipping.")
        continue

    # Prepare input tensors
    stock_seq = torch.tensor(window_data[['open', 'high', 'low', 'close', 'volume']].values, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get news vectors for the window
    news_vectors = [get_daily_news_vector(date) for date in window_data['date']]
    news_tensor = torch.tensor(np.array(news_vectors).flatten(), dtype=torch.float32).unsqueeze(0).to(device)

    # Make prediction
    model.eval()
    with torch.no_grad():
        pred_normalized = model(stock_seq, news_tensor).squeeze().item()

    # Inverse transform to get actual price
    scaler = scalers[company]
    dummy = np.zeros((1, 5))
    dummy[0, 3] = pred_normalized  # 'close' is at index 3
    pred_price = scaler.inverse_transform(dummy)[0, 3]

    # Get latest actual price for reference
    latest_date = window_data['date'].max()
    latest_price = window_data.loc[window_data['date'] == latest_date, 'close'].iloc[0]
    latest_dummy = np.zeros((1, 5))
    latest_dummy[0, 3] = latest_price
    latest_actual_price = scaler.inverse_transform(latest_dummy)[0, 3]
    
    # Calculate predicted change
    change = ((pred_price - latest_actual_price) / latest_actual_price) * 100
    direction = "UP ▲" if change > 0 else "DOWN ▼"
    
    print(f"{company}: Predicted close for {pred_date.strftime('%Y-%m-%d')} = ${pred_price:.2f}")
    print(f"  Previous close ({latest_date.strftime('%Y-%m-%d')}): ${latest_actual_price:.2f}")
    print(f"  Predicted change: {change:.2f}% {direction}\n")