# Milestone 2 Discussion: RAG Pipeline Evaluation

## 1. Model Choice

We used `llama3.1:8b` via Ollama because:

- Strong reasoning and summarization ability
- Runs locally without requiring cloud API access
- Supports tool calling for future extensions
- Performs well on structured RAG prompts

---

## 2. Prompt Template Experiments

We experimented with two system prompt variants:

**Prompt V1** — General shopping assistant, instructed to cite ASINs and be concise.
> Produced more concise answers but occasionally hallucinated product details.

**Prompt V2** — Expert appliance analyst, instructed to say "I don't know" if context is insufficient.
> More conservative and better at avoiding hallucinations, but sometimes less detailed.

**Selected:** Prompt V1 for final use due to better overall user experience.

---

## 3. Top-k Retrieval Experiments

| k | Observation |
|---|-------------|
| 3 | High precision, but sometimes missing useful context |
| 5 | Best balance — chosen for final pipeline |
| 10 | More complete context but introduced noise into answers |

---

## 4. Hybrid RAG Qualitative Evaluation

We ran the Hybrid RAG pipeline on 5 queries and manually rated each answer on three dimensions.

**Rating scale:** Out of 5, 5(perfect) and 1 (not correct at all)

| Query | Answer Summary | Accuracy(/5) | Completeness(/5) | Fluency(/5) |
|-------|---------------|--------------|------------------|-------------|
| washing machine not draining water | Correctly identified COMFEE' washer (B09W5PMK5X) as having a continuous draining defect, grounded in real reviews | 5 | 4 | 5 |
| temperature controlled fridge | Recommended two relevant products with correct ASINs and accurate feature descriptions from reviews | 5 | 4 | 5 |
| appliance to cool down my home | Recommended a skincare mini fridge and a dishwasher magnet — neither cools a home. Model correctly noted the context was insufficient but still gave a wrong suggestion | 1 | 1 | 4 |
| my microwave is too loud! | Suggested a corn/potato cooker as a quieter alternative — off-topic and unhelpful | 2 | 1 | 4 |
| my grandma loves to bake, what should i get for her birthday? | Recommended an egg holder and an ice maker — neither is baking-related | 1 | 1 | 4 |

---

## 5. Key Observations

The Hybrid RAG pipeline performs well when the query closely matches review content in the corpus (for example - washing machine not draining water). It struggles with queries that ask about products not well represented, or represented in a weird way (for example - appliance to cool down my home). In these cases, the LLM generates plausible-sounding but irrelevant answers rather than declining to respond.

---

## 6. Limitations

1. Small corpus size - The 1,149-review stratified sample covers a limited range of appliance types.
2. LLM hallucinating answers - rather than saying 'I don't know', the LLM gives non relevant answers.

---

## 7. Suggestions for Improvement

We could expand the corpus and add a web search tool (like Tavily) to supplement the corpus for queries about products not in the reviews.
