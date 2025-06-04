# CS145 Recommendation System Competition - Checkpoint 2: Sequence-Based Recommenders

## Overview

Checkpoint 2 focuses on **Sequence-Based Recommendation**. In a sequential setting, the order of user interactions matters: a user's next action is conditioned on what they have done before. You will design algorithms that leverage interaction histories (clicks, views, purchases) to recommend items that maximize **discounted revenue**.

## Why Sequence Matters

Many purchasing behaviors are temporally dependent. For example, users may browse cheap accessories before buying an expensive phone, or binge-watch episodes sequentially. A model that understands these patterns can surface higher-value items at the right moment.

## Task

1. **Curate Sequences**:
   - Maintain an ordered list of each user's historical interactions across training iterations.
   - Each element should minimally contain `(timestamp, item_id, price, response)`.
2. **Implement at least THREE sequence-aware recommenders** chosen from:
   - **Auto-Regressive (AR) Models** (e.g., n-gram autoregressive models)
   - **Recurrent Neural Networks (RNN/GRU)**
   - **Long Short-Term Memory (LSTM)**
   - **Transformer / Self-Attention architectures**
3. **Hyperparameter Tuning**: Experiment with different sequence lengths, embedding sizes, hidden units, regularization (dropout, L2, weight decay), and learning rates.
4. **Ranking & Revenue Optimization**: Output a score (logit/probability) per candidate item, then rank by:  
   `expected_revenue = price × probability`  
   Use this for top-`k` recommendation.
5. **Compare Performance** across all implemented models using the platform metrics.

## Data Handling Guidelines

1. **Sequence Construction**
   - Append new interactions after each training iteration.
   - For efficiency, cap the maximum sequence length (e.g., last 50–100 events) and use **padding** or **masking** during batch training.
2. **Feature Representation**
   - Item embeddings can be learned end-to-end or initialized from content-based features.
   - Incorporate **positional embeddings** for Transformer models.
3. **Mini-Batching**
   - Group sequences of similar lengths or apply dynamic padding with `torch.nn.utils.rnn.pack_padded_sequence`.

## Model Implementation Notes

### A. Auto-Regressive / Markov
- Treat the last `n` items as the state, predict the next item.
- Try orders 1, 2, 3 and smoothing schemes (add-k, back-off).

### A. Auto-Regressive (AR)
- Treat the last `n` items as context and predict the next item (n-gram autoregressive).
- Try orders 1, 2, 3 and experiment with additive smoothing or other regularization techniques.

### B. RNN / LSTM / GRU
- Key hyperparameters: hidden size (64, 128, 256), #layers (1–3), dropout (0.1–0.5).
- Use **embedding layers** for items and (optionally) prices or categories.
- Apply **teacher forcing** during training; during inference, feed the entire history and score candidate items.

### C. Transformer
- Base your implementation on **SASRec** or **BERT4Rec** style encoders.
- Hyperparameters: #heads (2–8), hidden size, feed-forward size, dropout, #blocks.
- Use **causal masking** to prevent peeking at future interactions.

## Regularization Techniques

- **Dropout** (input, hidden, embedding)
- **Layer normalization** (Transformers)
- **Weight decay / L2**
- **Early stopping** on validation discounted revenue
- **Gradient clipping** to stabilize RNN training

## Evaluation

Use the provided simulation to evaluate on:
1. **Primary**: Discounted Revenue
2. **Secondary**: Total Revenue, Precision@K, NDCG@K, MRR, Hit Rate

Plot learning curves and compare models under identical train/test splits.

## Allowed Tools and Libraries

You may use the following data science packages:
- pandas
- numpy
- scikit-learn
- pytorch (if you want to implement neural approaches)
- matplotlib/seaborn (for visualization)
- Any sklearn-compatible libraries (XGBoost, LightGBM, etc.)

## Deliverables

Be prepared to submit the following as part of the final deliverables:

1. **Code**: Your implementation in a well-organized Python module
2. **Report** containing:
   - Description of implemented approaches
   - Feature engineering techniques used
   - Hyperparameter tuning methodology and results
   - Performance comparison between different models
   - Analysis of what worked well and what didn't

3. **Presentation slides** (5-8 slides) summarizing your approach and results

## Grading Criteria (Tentative) 

- **Implementation correctness**: 40%
- **Model performance**: 30%
- **Approach diversity**: 15%
- **Report quality and insights**: 15%

## Getting Started

1. Clone the repository and set up the environment as described in the main README
2. Familiarize yourself with the `recommender_analysis_visualization.py` script
3. Start by implementing a simple content-based model as a baseline
4. Gradually improve your approach by adding more sophisticated methods
5. Use the provided evaluation framework to compare different approaches

Remember: The goal is to maximize **discounted revenue** by recommending items that users are likely to purchase at their given prices! 


## Mid-Project Check-In (Due 2025-06-04)

By **23:59 PT on 2025-06-04** upload a concise progress report (≈ 1–2 pages) that includes:

⦿ **Checkpoint 1 summary** – key results and insights from your content-based recommender.
⦿ **Checkpoint 2 progress** – current status of your sequence pipeline, models implemented, preliminary metrics, and challenges encountered.
⦿ **Next steps** – a short plan outlining what you will complete before the final deadline.

Submit the report (PDF) to Gradescope. One report per team.


> **Remember:** The winning algorithm will balance **sequence understanding** with **revenue-aware ranking** to surface the most valuable items at the right time. 