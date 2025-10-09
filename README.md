# trading-platform-test
Trading platform by Juan Lopez

Is it possible to use GenAI to perform trading decisions, knowing that LLM pipelines are not nowhere near under 0.5s inference times?


Index


1. Prototype

2. Systems Design


The experimentation and the results of prototype are the ones that motivated the proposed system design


# 1. Prototype

How to reconcile costly LLM calls with sub 0.500 inferences?

Our goal to speed-up costly agent computations is to use something called *semantic cache*. It is basically precomputating responses of prompts and saving them in cache memory, then, when a similar prompt enters the system, we match it with its response by finding the original prompt in a vector search. We will be using redisvl, which is the redis vector database. Since is Redis, this vector db will live in memory and not in disk.

This system is designed to be run via docker compose, by pulling the three different docker images. I set up the repo that implements the trading-api in its own repo (here: https://github.com/JCLMantilla/llm-cache), in that way I can do easy versioning by just using github actions to upload Docker images on release into the Github Container Registry. This approach is a simple yet powerful way of doing CI/CD. It is very handy wen We iterate from a `commit` -> `Build image` -> `Pull it in a Kubernetes yaml` in a very fast way.

*Important Note*

- We will be using openai embeddings small. Since they are fast and robust.
- For this prototype our goal is to prove that we can map input prompts from agents into their responses using semantic caching. This would serve as a POC to show that agentic workflows that might be optimized by replacing calls with cache lookups. A lot of research and experimentation should be put into that, to see if by using semantic caching we can have high quality decisions for a trading operation.
- For this project we will get a success caching ratio of 100% since we will use the same prompts to match to their responses (also, we will get only 1 match per lookup with a similariity threshold tau of 0.1). This has been done this way for the sake of simplicity over a tight deadline. 

Please go into the notebook.ipynb to see some benchmarks of speed and others


## Achieved implementation


We managed to get an implementation that uses full-prompt embeddding with SC lookup using a test dataset `df_events`. Further work have to put in order to implement the embedding cache lookup, and the vector composition via sum-norm.


Please create a .env file in the root directory with the following variables:
``` bash
APP_PORT=8006
OPENAI_API_KEY=your_api_key # I will provide you my own openai key if you dont have one ;)
```

run 

``` bash
docker compose up --build
```

It will pull:

- `redis image`
- `Prometheus image`
- `llm-cache` image from my github container registry


It will start:

*Trading API* (port 8005)
```bash
    http://localhost:8005
```
*Prometheus* (port 9090)
```bash
     http://localhost:9090/query
```

and Redis of course 

## Endpoints of Trading API:

For a more details visit de swagger docs: ```bash http://localhost:8005/docs ```


### `/clear`

 Clear all cache


### `/decide`

 Uses a string to perform a semantic cache look-up using Redis and openai embeddings (text-embedding-3-small). 
 This endpoint has an argument and its called `manual_embeddings` and when its True it records speed both of embeddings and cache lookup. However, this endpoint does not handle a lot of concurrent request (I could not implement my own embedding job handler as good as the Redis one).

 When its False, the endpoint uses the vectorizer in the same call as the lookup, it can handle a lot of concurrent requests but we cannot record embbeding time with it.



### `/insert_data`

 Inserts data in the same format as the df_events.parquet dataset. Take a look at the Pydantic schema in Swagger, there is an example you can use.


## Monitoring


The app includes comprehensive Prometheus metrics to monitor:
- HTTP request metrics (count, latency, status codes)
- Redis semantic cache operation metrics (Latency, and catch/miss rates)
- Application error tracking

### Metrics Endpoints

The main metrics endpoint `/metrics` exposes all application metrics in Prometheus format. These can be visualized in:

`http://localhost:8005/metrics`


To inspect the metrics using Prometheus dashboard: 


Access: `http://localhost:9090/query`


### Recommended queries for metrics and performances

1. **General Requests**

   - Errors by endpoint and status: `application_errors_total`

   - Requests by endpoint: `http_requests_total` (you can filter by endpoint e.g `http_requests_total{endpoint="/insert_data"}` or `http_requests_total{endpoint="/decide"}`)
   
   - Request rate by endpoint (in the past 5 mins): `rate(http_requests_total[5m])`

   - Latencies for /decide endpoint: `http_request_duration_seconds_bucket{endpoint="/decide"}`. This will show you a list of the number of requests that took ≤ 0.5 seconds (le="0.5"), and also  ≤ 0.75 seconds (le="0.75") etc. This latency includes embedding + caching duration + the rest of call-related operations

   - Average latency for decide endpoint: `http_request_duration_seconds_sum{endpoint="/decide"} / http_request_duration_seconds_count{endpoint="/decide"}`

   - p0.95 latency for decide endpoint: `histogram_quantile(0.95, http_request_duration_seconds_bucket{endpoint="/decide"})` 


2. **Redis Operations**

    Caching

   - Average latency for the caching operation: `semantic_cache_duration_seconds_sum / semantic_cache_duration_seconds_count`

   -  p0.95 latency for the caching operation: `histogram_quantile(0.95, semantic_cache_duration_seconds_bucket)` 

   - Number of hits: `semantic_cache_hits_total`

   - Number of misses: `semantic_cache_misses_total`
   
   - Cache global hit ratio: `semantic_cache_hits_total/semantic_cache_hits_total + semantic_cache_misses_total`

   - Cache global miss ratio: `semantic_cache_misses_total/semantic_cache_hits_total + semantic_cache_misses_total`


   Embeddings

   - Average latency for the caching operation: `openai_embedding_duration_seconds_sum / openai_embedding_duration_seconds_count`

   - p0.95 letency for embedding operation: `histogram_quantile(0.95,openai_embedding_duration_seconds_bucket)`



    Dedicated metrics for storing

    - All inserting operations: `redis_insert_operations_total`

    - Latency for storing operations: `redis_insert_duration_seconds_bucket{operation="astore"}`

    - Latency percentile of inserts: `histogram_quantile(0.95, rate(redis_insert_duration_seconds_bucket{operation="astore"}[20h]))`

## Results

After inserting the first 50 records of the df_events dataset and performing a semantic cache with the first 30 records we have the following benchmarks:


### Average embedding duration: 

`openai_embedding_duration_seconds_sum / openai_embedding_duration_seconds_count`

```Python
{instance="trading-api-container:8005", job="trading-platform", operation="embedding"}  	0.47188677787780764
```

### p0.95 latency for embeddings: 

`histogram_quantile(0.95,openai_embedding_duration_seconds_bucket)`

```Python
{instance="trading-api-container:8005", job="trading-platform", operation="embedding"}	    0.7187499999999999
```

### Average latency for semantic cache match: 

`semantic_cache_duration_seconds_sum / semantic_cache_duration_seconds_count`

```Python
{instance="trading-api-container:8005", job="trading-platform", operation="acheck"}	        0.004074819882710775
```
### p0.95 latency for semantic cache: 

`histogram_quantile(0.95, semantic_cache_duration_seconds_bucket)` 

```Python
{instance="trading-api-container:8005", job="trading-platform", operation="acheck"}	        0.009375

```

## Now lets dive into the actual benchmarks of the endpoint:

### Average latency: 

`http_request_duration_seconds_sum{endpoint="/decide"} / http_request_duration_seconds_count{endpoint="/decide"}`

```Python
{endpoint="/decide", instance="trading-api-container:8005", job="trading-platform", method="POST"}	    0.4941580216089884

```

### p0.95 Latency: 

`histogram_quantile(0.95, http_request_duration_seconds_bucket{endpoint="/decide"})`

```Python
{endpoint="/decide", instance="trading-api-container:8005", job="trading-platform", method="POST"}	0.7249999999999999
```


## *We can see that that the main thing holding the speed of the program is the embedding time!*

Then, how can we can achieve sub .5s decisions? by speeding up embeddings or precomputating them.



# 2. System Design

As we discovered in the results of the prototype, embedding time is the main bottle neck of the system. Then, how can we can achieve sub .5s decisions? 

*by speeding up embeddings or precomputating them*

## Goals

- Achieve <500ms p95 decisions by removing full-prompt embedding from the hot path
- Keep accuracy high by reusing rich, pre-analyzed context (instructions, frameworks, historical news, examples)

## How to do it?

- Maintain two caches:
  - Embedding Cache (EC): pre-embedded, frequently reused assets (instruction prompts, tool schemas, market context templates, prior/analyzed news chunks, etc.)
  - Semantic Cache (SC): keys are composed vectors representing (instruction + context + recent-news) and values are the final decision payloads (signal, conviction, reasoning, metadata).
- At runtime, only embed the small new piece (incoming news chunk). Compose its vector with pre-embedded assets via sum+normalize to approximate the full prompt embedding. Query SC. If a high-similarity hit exists, return its decision immediately. Otherwise, run full prompt embeddings to do the SC match or run the LLM agents to take a decision (if we allow it) and to backfill SC for future cahce lookups.

## Data model (Redis)

- Embedding Cache (*EC*) — Redis hashes (works with `redisvl` embeddings cache):
  - Key patterns:
    - `ec:prompt:{id}` for instruction/system prompts
    - `ec:news:{id}` for pre-analyzed or historically relevant news chunks
  - Fields:
    - `vector` (base64 of float32), `dim`, `model`, `text_hash`, `created_at`, `ttl_sec`, `meta` (JSON: type, topic, sector, tags, timeframe, etc.)

- Semantic Cache (*SC*) — RediSearch index over vector field using HNSW:
  - Key pattern: `sc:combo:{sha}` where `sha` is SHA256 over sorted component IDs + model + strategy
  - Fields:
    - `vector` (composed float32 or base64), `dim`, `model`, `strategy`, `tau` (threshold used), `vec_details` (JSON list of `{type,id,weight}` and other info of the composed vector),
    - `decision` (JSON: `signal`, `conviction`, `reasoning`, `agent_versions`, `latency`, `labels`),
    - `created_at`, `ttl_sec`
  - RediSearch index config (HNSW) if needed

## Vector composition

- We want compose vectors V = normalize(sum(w_i * v_i)) where `v_i` are component embeddings (instruction, framework prompts, pre-embedded context, runtime news), and weights `w_i` are tuned per strategy (e.g., instruction=0.4, examples=0.2, historical_context=0.2, runtime_news=0.2). We need to start with equal weights, then learn with offline evals.
- We could have another kind of composed vector, which is the direct concatenated vector with reduced dimentionality (via PCA or other technique) for improved separability and richer representation.

## Fast-> slow -> agentic path algorithm

1) Ingestion (async, continuous):
   - Chunk and embed: instruction prompts, frameworks, tool specs, exemplars, and historical or analyst-curated news → store in EC.
2) Request `/decide` (runtime):
   - Step A: Embed only the incoming news chunk (use small, fast model; cap to N tokens; cache by `text_hash`).
   - Step B: Fetch K relevant EC items by key or quick metadata lookup (e.g., same tag/domain, location or latest time frame). Typically instruction + 1-4 context chunks + runtime news.
   - Step C: Compose vector via sum+normalize. If top-1 similarity ≥ τ, return cached `decision` immediately (<50ms typical after single embedding since cache lookups are extremely fast).
   - Step D: If miss or below τ:
       -  slow path: concat runtime news + pre embeded chunks + other news that might be good candidates for decision (similarity difference close to τ), and check if it can trigger a decision. Note that requests that embed a full prompt for SC takes less than 1 secs compared with 2-5s Agentic runs.
       - Trigger agentic-path multi-agent (News Sentiment, Market Context, Risk) in parallel. Aggregate decisions (weighted voting or learn-to-rank of agent outputs). Produce final decision.

    - Step F: 
        - Backfill SC with the composed vector + `decision` if it was calculated before by agents in step D by agents. After the agentic path produces a decision (2-5s), insert a new item into the Semantic Cache where the key’s vector is the composed vector (instruction + context + runtime news) and the value is the final decision payload. Future similar requests will hit this entry instead of recomputing
        - Store the runtime-news embedding in EC for future reuse.
    

## Thresholds and budgets

- Similarity threshold `τ` start at 0.86–0.92 (cosine); tune by offline evals to achieve ≥80% hit-rate p95.
- Latency budget (p95 targets):
  - Runtime news embedding: 120–180ms (small model)
  - EC fetch + compose: 5–15ms
  - SC vector search: 5–25ms
  - Decision serialization + response: 10–30ms
  - Total fast path: 160–250ms (p95)

  if slow + agentic path allowed to decide:
  
  - Slow path: 500 ms aprox based on prototype results
  - Agentic path adds up to 2-5s based on your reports
  
## Chunking strategy

- Pre-embedded context: 256–512 token chunks with 64-token overlap (overlap to be tuned).
- Runtime news: cap at 256 tokens for speed. We can preprocess it using regex rules and/or a domain-specific summarizer (small local model) before embedding.

## Multi-agent orchestration

- Run three agents in parallel with shared retrieved context.
- Aggregation: majority vote on `signal`, and avoid ties based on `conviction` level of each signal. We can also use a small classifier trained offline to map agent features + additional indicators to final signal.
- Add per-agent timeouts (e.g., 800ms soft, 1500ms hard) and circuit breakers. Maybe fall back to smaller local models when timeouts hits or run them im parallel and take decision based on which result arrived first.

## Monitoring

- Prometheus (already implemented): For metrics for EC hits/misses, SC hits/misses, similarity scores distribution, composition weights, and per-step latencies, etc.
- Periodic offline eval to tune weights for vector composition of embeddins `w_i`, `τ` and decision system based on actual behavior of bitcoin vs agentic responses


## Minimal pseudocode

```python
def decide(request_news_text):
    news_vec = embed_small(request_news_text)               #  <200ms 
    instr_vec = ec_get('ec:prompt:instruction_v3')          # ~1ms
    ctx_vecs = ec_get_latest_context(tags, locatios, N=2)   # ~2–5ms
    composed = normalize(sum([instr_vec, *ctx_vecs, news_vec]))
    hit = sc_search_top1(composed)
    if hit and hit.similarity >= tau:
        return hit.decision, meta(hit)
    decision = run_agents_parallel(request_news_text)       # async/agentic path -> Only if alowed to take a decision
    sc_backfill(composed, decision)
    ec_store(news_vec)
    return decision, metadata 
```

This design hopefuly removes then need for full-prompt embedding from the decision path, since converts most requests into etremely fast cache lookups, and uses agentic path only when allowed and to to backfill SV.


# 3. Databricks pipeline


This pipeline will be provided soon.

