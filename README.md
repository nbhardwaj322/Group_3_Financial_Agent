# Group_3_Financial_Agent
An AI-powered finance agent that integrates historical stock data, market trend analysis, and real-time news to generate clear buy or not-buy recommendations for intelligent investment decisions.


## Usage
1. Set the target stock ticker symbol in the execution script.
2. Run the end-to-end analysis pipeline:
   - Plan research steps.
   - Download and analyze historical market data.
   - Fetch and summarize news.
   - Generate investment recommendations.
   - Store recommendation in memory.
   - Evaluate recommendation quality.

## Example
The provided code runs an example for the ticker symbol "AAPL" to demonstrate full pipeline execution, including market trend extraction, relevant news retrieval, summarization, and evaluation.

## Technologies Used
- yfinance: Historical stock data acquisition
- feedparser: Financial news RSS parsing
- Transformers (Flan-T5): Summarization and text generation
- SentenceTransformers & FAISS: Semantic embeddings and memory indexing
- pandas, numpy: Data handling and numerical operations

## Project Structure
- `PlannerAgent`: Research planning component
- `DataAgent`: Historical data downloader
- `AnalysisAgent`: Market trend analyzer
- `NewsAgent`: News retriever and summarizer
- `SummaryAgent`: Recommendation generator using LLM
- `MemoryAgent`: Embedding and storage of past insights
- `EvaluatorAgent`: Recommendation quality evaluation

## Future Work
- Integrate real-time streaming news and market data APIs for enhanced responsiveness.
- Expand NLP capabilities to include detailed sentiment analysis and explainable insights.
- Build a conversational interface to enable user interaction with the agent.
- Add backtesting and portfolio management modules.
