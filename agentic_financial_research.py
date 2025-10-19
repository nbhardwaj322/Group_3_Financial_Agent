# Agentic Financial Research Agent Notebook
# Full step-by-step implementation for Apple and Tesla research using an agentic AI

# CELL 1: Install dependencies
!pip install --upgrade pip
!pip install yfinance transformers sentence-transformers faiss-cpu

# CELL 2: Imports
import yfinance as yf
import pandas as pd
import numpy as np
import faiss
from transformers import pipeline
from sentence_transformers import SentenceTransformer

# CELL 3: Agent Setup
class InvestmentResearchAgent:
    def __init__(self):
        print("Initializing agent...")
        self.llm = pipeline("text2text-generation", model="google/flan-t5-base")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = faiss.IndexFlatL2(384)
        self.memory = {}

    # Step 1: Plan research
    def plan_research(self, ticker):
        steps = [
            f"Collect 1-year historical data for {ticker}",
            f"Compute price and volume trends for {ticker}",
            f"Generate short-term outlook and evaluation summary for {ticker}"
        ]
        return steps

    # Step 2: Retrieve Yahoo Finance data
    def get_yahoo_data(self, ticker):
        df = yf.download(ticker, period="1y", interval="1d", progress=False)
        if df is None or df.empty:
            print(f"⚠ Warning: No data found for {ticker}")
            return pd.DataFrame()
        return df.tail(10)

    # Step 3: Analyze market trend
    def analyze_market_trend(self, df):
        if df.empty or 'Close' not in df.columns or df['Close'].dropna().empty:
            return 'No Data', 0.0, 0.0
        daily_change = df['Close'].pct_change().dropna()
        if daily_change.empty:
            return 'No Data', 0.0, 0.0
        change = daily_change.mean() * 100
        volatility = daily_change.std() * 100
        trend = 'uptrend' if change > 0 else 'downtrend'
        return trend, round(change, 2), round(volatility, 2)

    # Step 4: Summarize findings
    def summarize_findings(self, ticker, trend, change, volatility):
        prompt = (
            f"The stock {ticker} shows a {trend} with an average daily change of {change}% "
            f"and volatility of {volatility}%. Provide a short investment insight in 3 lines."
        )
        response = self.llm(prompt, max_new_tokens=80)[0]['generated_text']
        return response.strip()

    # Step 5: Evaluate summary
    def evaluate_summary(self, summary):
        prompt = f"Evaluate the clarity and usefulness of this insight: '{summary}' in one line."
        feedback = self.llm(prompt, max_new_tokens=50)[0]['generated_text']
        return feedback.strip()

    # Step 6: Store memory
    def store_memory(self, ticker, text):
        embedding = self.embedder.encode([text])
        self.index.add(np.array(embedding, dtype='float32'))
        self.memory[ticker] = text

    # Step 7: Full run
    def run(self, ticker):
        print(f"\n===== Running Research for {ticker} =====")
        steps = self.plan_research(ticker)
        print("Research Plan:")
        for s in steps:
            print(" -", s)

        df = self.get_yahoo_data(ticker)
        trend, change, volatility = self.analyze_market_trend(df)
        if trend == 'No Data':
            print(f"⚠ Skipping {ticker} due to missing data")
            return

        summary = self.summarize_findings(ticker, trend, change, volatility)
        feedback = self.evaluate_summary(summary)
        self.store_memory(ticker, summary)

        print("\n📈 Analysis Result:")
        print(f"Trend: {trend}, Avg Change: {change}%, Volatility: {volatility}%")
        print(f"\n🧠 Insight:\n{summary}")
        print(f"\n💬 Evaluator Feedback:\n{feedback}")
        print("-"*80)

# CELL 4: Test Full Run
agent = InvestmentResearchAgent()
tickers = ["AAPL", "TSLA"]
for ticker in tickers:
    agent.run(ticker)

print("\n✅ Research completed for Apple and Tesla.")