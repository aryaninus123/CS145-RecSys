from __future__ import annotations
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd
from pyspark.sql import DataFrame, SparkSession, functions as sf
from sim4rec.utils import pandas_to_spark

__all__ = ["ARRecommender"]

class ARRecommender:
    def __init__(
        self,
        order: int = 1,          # bigram by default
        alpha: float = 10.0,     # stronger smoothing
        max_sequence: int = 100,
        user_id_col: str = "user_idx",
        item_id_col: str = "item_idx",
        debug: bool = False,
    ) -> None:
        if order < 1:
            raise ValueError("order must be ≥1")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.order = order
        self.alpha = alpha
        self.max_sequence = max_sequence
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.debug = debug

        self.item_prices: Dict[int, float] = {}
        self.total_items: int = 0
        self.user_histories: Dict[int, List[int]] = defaultdict(list)
        self.ngram_counts: Dict[Tuple[int, ...], Counter[int]] = defaultdict(Counter)
        self.unigram_counts: Counter[int] = Counter()

    def _update_prices(self, items_df: DataFrame) -> None:
        for row in items_df.select(self.item_id_col, "price").toPandas().itertuples(index=False):
            self.item_prices[getattr(row, self.item_id_col)] = row.price
        self.total_items = len(self.item_prices)

    def _add_alpha_probs(self, counter: Counter[int]) -> Dict[int, float]:
        V = self.total_items or 1
        denom = sum(counter.values()) + self.alpha * V
        base = self.alpha / denom
        probs = {it: base for it in self.item_prices}
        for it, cnt in counter.items():
            probs[it] = (cnt + self.alpha) / denom
        return probs

    def _context_probs(self, context: Tuple[int, ...]) -> Dict[int, float]:
        for L in range(len(context), 0, -1):
            c = context[-L:]
            if c in self.ngram_counts:
                return self._add_alpha_probs(self.ngram_counts[c])
        return self._add_alpha_probs(self.unigram_counts)

    def fit(self, log: DataFrame | None, user_features=None, item_features=None):
        if log is None or log.count() == 0:
            return
        if not self.item_prices and item_features is not None:
            self._update_prices(item_features)
        if self.total_items == 0:
            return

        pdf = log.select(self.user_id_col, self.item_id_col, "relevance").toPandas()
        pdf["relevance"] = pdf["relevance"].gt(0).astype(int)

        if self.debug:
            print(f"[AR] fit: processing {len(pdf)} interactions (order={self.order})")

        for uid, iid, rel in pdf.itertuples(index=False):
            price = self.item_prices.get(iid, 0.0)
            weight = self.item_prices.get(iid, 0.0) * (1 if rel == 1 else 0.2)

            hist = self.user_histories[uid]
            if hist and weight > 0:
                ctx = tuple(hist[-self.order:])
                self.ngram_counts[ctx][iid] += weight
            if weight > 0:
                self.unigram_counts[iid] += weight
            # push to history regardless of relevance
            hist.append(iid)
            if len(hist) > self.max_sequence:
                self.user_histories[uid] = hist[-self.max_sequence:]

        if self.debug:
            print(f"[AR] after fit: users={len(self.user_histories)} | contexts={len(self.ngram_counts)} | items={self.total_items}")

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame, **kwargs):

        spark = SparkSession.builder.getOrCreate()
        filter_seen = kwargs.get("filter_seen_items", True)
        if len(self.unigram_counts) < 1000:
            filter_seen = False    # let early repeat purchases happen

        seen = defaultdict(set)
        if filter_seen and log is not None and log.count() > 0:
            for row in log.select(self.user_id_col, self.item_id_col).toPandas().itertuples(index=False):
                seen[getattr(row, self.user_id_col)].add(getattr(row, self.item_id_col))

        # Pre‑compute global popularity prob for fallback
        pop_probs = self._add_alpha_probs(self.unigram_counts)

        results, top1 = [], []
        user_ids = users.select(self.user_id_col).distinct().toPandas()[self.user_id_col]
        for uid in user_ids:
            hist = self.user_histories.get(uid)
            if not hist:
                context_scores = None
            else:
                context_scores = self._context_probs(tuple(hist[-self.order:]))

            # If context very sparse, blend 0.3*popularity
            if context_scores is None:
                blended = pop_probs
            else:
                blended = {it: 0.7*context_scores[it] + 0.3*pop_probs[it] for it in self.item_prices}

            scores = {it: blended[it]*self.item_prices[it] for it in blended}
            if filter_seen:
                for it in seen[uid]:
                    scores.pop(it, None)
            if not scores:
                continue
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
            top1.append(ranked[0][1])
            for it,_ in ranked:
                results.append({self.user_id_col: uid, self.item_id_col: it, "relevance": blended[it]})

        if self.debug:
            n_users = len(top1)
            mean_rev = sum(top1)/n_users if n_users else 0
            print(f"[AR] predict: users_out={n_users}  mean_top1_expected_rev={mean_rev:.2f}  total_rows={len(results)}")

        pdf = pd.DataFrame(results, columns=[self.user_id_col, self.item_id_col, "relevance"])
        spark = SparkSession.builder.getOrCreate()
        if pdf.empty:
            from pyspark.sql.types import StructType, StructField, IntegerType
            schema = StructType([
                StructField(self.user_id_col, IntegerType(), False),
                StructField(self.item_id_col, IntegerType(), False),
                StructField("relevance", IntegerType(), False),
            ])
            return spark.createDataFrame([], schema)

        sdf = pandas_to_spark(pdf, spark_session=spark)
        sdf = sdf.withColumn(self.user_id_col, sf.col(self.user_id_col).cast("int"))
        sdf = sdf.withColumn(self.item_id_col, sf.col(self.item_id_col).cast("int"))
        sdf = sdf.withColumn("relevance", sf.round(sf.col("relevance")).cast("int"))
        return sdf
