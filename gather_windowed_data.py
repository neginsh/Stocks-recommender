import yfinance as yf


ticker_list = [
    # Technology
    "AAPL", "MSFT", "GOOG", "AMZN", "META", "NVDA", "TSLA", "ORCL", "IBM", "ADBE",
    
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "BLK", "SCHW", "PYPL",
    
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "ABBV", "LLY", "BMY", "TMO", "GILD", "AMGN",
    
    # Energy
    "XOM", "CVX", "BP", "SHEL", "COP", "SLB", "TOT", "ENB", "EQNR", "PSX",
    
    # Consumer & Retail
    "WMT", "PG", "KO", "PEP", "MCD", "NKE", "COST", "DIS", "HD", "SBUX"
]


def get_data():
    df = yf.download(ticker_list, period='5y', progress=False)
    df = df.rename(columns={"Close": "close", "Open": "open", "High": "high", "Low": "low", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    print("Loaded data from yfinance:", df.shape)
    return df

def split_train_test(df):
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)
    return train_df,test_df
