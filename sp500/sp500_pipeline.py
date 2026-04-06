"""
S&P 500 Data Pipeline
Downloads 10 years of prices, computes returns/covariance, saves locally.

Requirements: pip install yfinance pandas numpy tqdm pyarrow
"""

import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
import sys

warnings.filterwarnings("ignore")


class SP500Pipeline:

    # S&P 500 tickers as of April 2026 (Wikipedia)
    TICKERS = [
        "MMM", "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A",
        "APD", "ABNB", "AKAM", "ALB", "ARE", "ALGN", "ALLE", "LNT", "ALL", "GOOGL",
        "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT", "AWK",
        "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT",
        "APP", "APTV", "ACGL", "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO",
        "ADSK", "ADP", "AZO", "AVB", "AVY", "AXON", "BKR", "BALL", "BAC", "BAX",
        "BDX", "BRK-B", "BBY", "TECH", "BIIB", "BLK", "BX", "XYZ", "BK", "BA",
        "BKNG", "BSX", "BMY", "AVGO", "BR", "BRO", "BF-B", "BLDR", "BG", "BXP",
        "CHRW", "CDNS", "CPT", "CPB", "COF", "CAH", "CCL", "CARR", "CVNA", "CAT",
        "CBOE", "CBRE", "CDW", "COR", "CNC", "CNP", "CF", "CRL", "SCHW", "CHTR",
        "CVX", "CMG", "CB", "CHD", "CIEN", "CI", "CINF", "CTAS", "CSCO", "C",
        "CFG", "CLX", "CME", "CMS", "KO", "CTSH", "COHR", "COIN", "CL", "CMCSA",
        "FIX", "CAG", "COP", "ED", "STZ", "CEG", "COO", "CPRT", "GLW", "CPAY",
        "CTVA", "CSGP", "COST", "CTRA", "CRH", "CRWD", "CCI", "CSX", "CMI", "CVS",
        "DHR", "DRI", "DDOG", "DVA", "DECK", "DE", "DELL", "DAL", "DVN", "DXCM",
        "FANG", "DLR", "DG", "DLTR", "D", "DPZ", "DASH", "DOV", "DOW", "DHI",
        "DTE", "DUK", "DD", "ETN", "EBAY", "SATS", "ECL", "EIX", "EW", "EA",
        "ELV", "EME", "EMR", "ETR", "EOG", "EPAM", "EQT", "EFX", "EQIX", "EQR",
        "ERIE", "ESS", "EL", "EG", "EVRG", "ES", "EXC", "EXE", "EXPE", "EXPD",
        "EXR", "XOM", "FFIV", "FDS", "FICO", "FAST", "FRT", "FDX", "FIS", "FITB",
        "FSLR", "FE", "FISV", "F", "FTNT", "FTV", "FOXA", "FOX", "BEN", "FCX",
        "GRMN", "IT", "GE", "GEHC", "GEV", "GEN", "GNRC", "GD", "GIS", "GM",
        "GPC", "GILD", "GPN", "GL", "GDDY", "GS", "HAL", "HIG", "HAS", "HCA",
        "DOC", "HSIC", "HSY", "HPE", "HLT", "HOLX", "HD", "HON", "HRL", "HST",
        "HWM", "HPQ", "HUBB", "HUM", "HBAN", "HII", "IBM", "IEX", "IDXX", "ITW",
        "INCY", "IR", "PODD", "INTC", "IBKR", "ICE", "IFF", "IP", "INTU", "ISRG",
        "IVZ", "INVH", "IQV", "IRM", "JBHT", "JBL", "JKHY", "J", "JNJ", "JCI",
        "JPM", "KVUE", "KDP", "KEY", "KEYS", "KMB", "KIM", "KMI", "KKR", "KLAC",
        "KHC", "KR", "LHX", "LH", "LRCX", "LVS", "LDOS", "LEN", "LII", "LLY",
        "LIN", "LYV", "LMT", "L", "LOW", "LULU", "LITE", "LYB", "MTB", "MPC",
        "MAR", "MRSH", "MLM", "MAS", "MA", "MKC", "MCD", "MCK", "MDT", "MRK",
        "META", "MET", "MTD", "MGM", "MCHP", "MU", "MSFT", "MAA", "MRNA", "TAP",
        "MDLZ", "MPWR", "MNST", "MCO", "MS", "MOS", "MSI", "MSCI", "NDAQ", "NTAP",
        "NFLX", "NEM", "NWSA", "NWS", "NEE", "NKE", "NI", "NDSN", "NSC", "NTRS",
        "NOC", "NCLH", "NRG", "NUE", "NVDA", "NVR", "NXPI", "ORLY", "OXY", "ODFL",
        "OMC", "ON", "OKE", "ORCL", "OTIS", "PCAR", "PKG", "PLTR", "PANW", "PSKY",
        "PH", "PAYX", "PYPL", "PNR", "PEP", "PFE", "PCG", "PM", "PSX", "PNW",
        "PNC", "POOL", "PPG", "PPL", "PFG", "PG", "PGR", "PLD", "PRU", "PEG",
        "PTC", "PSA", "PHM", "PWR", "QCOM", "DGX", "Q", "RL", "RJF", "RTX",
        "O", "REG", "REGN", "RF", "RSG", "RMD", "RVTY", "HOOD", "ROK", "ROL",
        "ROP", "ROST", "RCL", "SPGI", "CRM", "SNDK", "SBAC", "SLB", "STX", "SRE",
        "NOW", "SHW", "SPG", "SWKS", "SJM", "SW", "SNA", "SOLV", "SO", "LUV",
        "SWK", "SBUX", "STT", "STLD", "STE", "SYK", "SMCI", "SYF", "SNPS", "SYY",
        "TMUS", "TROW", "TTWO", "TPR", "TRGP", "TGT", "TEL", "TDY", "TER", "TSLA",
        "TXN", "TPL", "TXT", "TMO", "TJX", "TKO", "TTD", "TSCO", "TT", "TDG",
        "TRV", "TRMB", "TFC", "TYL", "TSN", "USB", "UBER", "UDR", "ULTA", "UNP",
        "UAL", "UPS", "URI", "UNH", "UHS", "VLO", "VTR", "VLTO", "VRSN", "VRSK",
        "VZ", "VRTX", "VRT", "VTRS", "VICI", "V", "VST", "VMC", "WRB", "GWW",
        "WAB", "WMT", "DIS", "WBD", "WM", "WAT", "WEC", "WFC", "WELL", "WST",
        "WDC", "WY", "WSM", "WMB", "WTW", "WDAY", "WYNN", "XEL", "XYL", "YUM",
        "ZBRA", "ZBH", "ZTS",
    ]

    def __init__(self):
        self.base = Path(__file__).parent
        self.raw = self.base/f"raw"
        self.proc = self.base/f"processed"
        self.end = datetime.today().strftime("%Y-%m-%d")
        self.start = (datetime.today() - timedelta(days=365*10+3)).strftime("%Y-%m-%d")
        self.batch_size = 100
        self.delay = 3.0

        for d in [self.raw, self.proc]:
            d.mkdir(parents=True, exist_ok=True)


    def get_tickers(self):
        print(f"{len(self.TICKERS)} tickers loaded")
        return self.TICKERS


    def download_prices(self, tickers):
        '''downloads adjusted close prices for the given tickers in batches (will download 492/503
        since some tickers are new or delisted within the 10 year window) '''
        batches = [tickers[i:i + self.batch_size] for i in range(0, len(tickers), self.batch_size)]
        print(f"\ndownloading {len(tickers)} tickers in {len(batches)} batches")

        frames = []

        for i, batch in enumerate(tqdm(batches, unit="batch")):
            path = self.raw/f"batch_{i:02d}.parquet"

            if path.exists():
                frames.append(pd.read_parquet(path))
                continue

            try:
                df = yf.download(batch, start=self.start, end=self.end, progress=False)
                df["Close"].to_parquet(path)
                frames.append(df["Close"])

            except Exception as e:
                tqdm.write(f"Batch {i:02d} failed: {e}")

            if i < len(batches) - 1:
                time.sleep(self.delay)

        prices = pd.concat(frames, axis=1)
        prices = prices.loc[:, ~prices.columns.duplicated()]
        prices.dropna(axis=1, how="all", inplace=True)
        print(f"price matrix: {prices.shape[0]} days x {prices.shape[1]} tickers")

        return prices


    def compute_matrices(self, prices):
        print("\ncomputing return & covariance matrices")
        prices.to_parquet(self.proc/f"prices_adj_close.parquet")

        # daily log returns: log(P_t / P_{t-1})
        daily = np.log(prices/prices.shift(1))
        daily = daily.dropna(how="all")

        # monthly log returns: resample to month-end, then log returns
        monthly_prices = prices.resample("ME").last()
        monthly = np.log(monthly_prices/monthly_prices.shift(1))
        monthly = monthly.dropna(how="all")

        daily.to_parquet(self.proc/f"returns_daily.parquet")
        monthly.to_parquet(self.proc/f"returns_monthly.parquet")

        daily_filtered = daily.loc[:, daily.notna().sum() >= 252]
        cov_daily = daily_filtered.cov() * 252 # 252 trading days in a year to annualize covariance
        cov_daily.to_parquet(self.proc/f"covariance_daily.parquet")

        monthly_filtered = monthly.loc[:, monthly.notna().sum() >= 24]
        cov_monthly = monthly_filtered.cov() * 12 
        cov_monthly.to_parquet(self.proc/f"covariance_monthly.parquet")


    def download_benchmark(self):
        print("\ndownloading SPY benchmark")

        spy = yf.download("SPY", start=self.start, end=self.end, progress=False)
        spy = spy[["Close"]].rename(columns={"Close": "SPY"})
        spy.to_parquet(self.proc/f"benchmark_spy.parquet")


    def run(self):
        tickers = self.get_tickers()
        prices = self.download_prices(tickers)
        self.compute_matrices(prices)
        self.download_benchmark()
    
        print(f"\ndone {self.base.resolve()}")


    def test(self):
        '''downloads a small test set of 5 tickers over the past year, computes matrices, saves locally'''
        test_tickers = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM"]
        test_start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
        test_raw = self.base/f"test/raw"
        test_proc = self.base/f"test/processed"

        for d in [test_raw, test_proc]:
            d.mkdir(parents=True, exist_ok=True)

        df = yf.download(test_tickers, start=test_start, end=self.end, progress=False)
        prices = df["Close"]
        print(f"Price matrix: {prices.shape} — missing: {prices.isna().sum().sum()}")

        prices.to_parquet(test_raw/"batch_00.parquet")
        print(f"Saved raw prices to {test_raw/'batch_00.parquet'}")

        daily = np.log(prices/prices.shift(1)).dropna(how="all") # daily log returns
        cov = daily.cov() * 252 # 252 trading days in a year to annualize covariance

        print(f"Daily returns: {daily.shape}")
        print(f"Cov matrix ({cov.shape}):\n{cov.round(4)}")

        prices.to_parquet(test_proc/f"prices_adj_close.parquet")
        daily.to_parquet(test_proc/f"returns_daily.parquet")
        cov.to_parquet(test_proc/f"covariance_daily.parquet")

        spy = yf.download("SPY", start=test_start, end=self.end, progress=False)
        spy[["Close"]].rename(columns={"Close": "SPY"}).to_parquet(test_proc/f"benchmark_spy.parquet")
        print(f"SPY rows: {len(spy)} saved to {test_proc/f'benchmark_spy.parquet'}")
  

if __name__ == "__main__":
    
    pipeline = SP500Pipeline()
    if "--test" in sys.argv:
        pipeline.test()
    else:
        pipeline.run()
