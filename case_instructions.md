# LLM Trading — Systems & Architecture

## Objective
Design and implement a near-real-time trading decision system that leverages LLMs for market analysis but still achieves sub-500ms decisions.

---

## Scenario
We are tasked with building a trading system with these constraints:

- **Data Ingestion**: Market news from 5 providers (Reuters, Bloomberg, AP, WSJ, FT).
- **LLM Agents**:
  - **News Sentiment Agent** → analyzes breaking news.  
  - **Market Context Agent** → evaluates current market conditions.  
  - **Risk Assessment Agent** → scores trade risk/reward.
- **LLM Latency**: Each inference takes 2–5 seconds.  
- **Decision Latency**: System must respond in <500ms (p95).  
- **Scale**: 100+ articles per minute during market hours.

**Key Challenge**  
How do you achieve <500ms decisions despite slow LLM inference?

**Data sources available** 
both for a single month (march)

event_df: (Contains processed reports from news by an llm to suggest a signal)
  pk / ulid: Unique identifiers for each record.
  schema_created_at / schema_upserted_at: Timestamps showing when the schema was created and last updated (all in mid-August 2025).
  datetime_est: The actual event timestamp (all on March 1, 2025, at different times).
  strategy_name: In this case, all entries use the strategy events_impacts_v1_vp.
  input_report / output_report: Contain descriptions of the context (inputs) and structured summaries (outputs) in JSON-like format.
  causality: Explain the reasoning process of the llm
  signal: 1, buy, 0 no trade, -1 sell
  conviction: how sure the llm is of its suggestion

market_df: (Cointains market information and prices)

