from abc import ABC
from typing import List, Union, Dict, Optional

import numpy as np
from scipy.stats import kstest
# TL;DR scipy.special is a C library, pylint needs python source code
# https://github.com/pylint-dev/pylint/issues/3703
# pylint: disable=no-name-in-module
from scipy.special import kl_div

from pyspark.sql import DataFrame
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    RegressionEvaluator,
    MulticlassClassificationEvaluator
)

from sdmetrics.reports.single_table import QualityReport


def evaluate_synthetic(
    synth_df : DataFrame,
    real_df : DataFrame
) -> dict:
    """
    Evaluates the quality of synthetic data against real. The following
    metrics will be calculated:

    - Column Shapes: Evaluates how well the synthetic data matches the shape of the real data
    - Column Pair Trends: Evaluates how well the synthetic data captures relationships between columns
    - Properties: Evaluates statistical properties of the synthetic data

    :param synth_df: Synthetic data without any identifiers
    :param real_df: Real data without any identifiers
    :return: Dictionary with metrics on synthetic data quality
    """

    report = QualityReport()
    report.generate(
        real_data=real_df.toPandas(),
        synthetic_data=synth_df.toPandas()
    )

    metrics = report.get_metrics()
    return {
        'column_shapes': metrics['Column Shapes']['Score'],
        'column_pair_trends': metrics['Column Pair Trends']['Score'],
        'properties': metrics['Properties']['Score']
    }


def ks_test(
    df : DataFrame,
    predCol : str,
    labelCol : str
) -> float:
    """
    Kolmogorov-Smirnov test on two dataframe columns

    :param df: Dataframe with two target columns
    :param predCol: Column name with values to test
    :param labelCol: Column name with values to test against
    :return: Result of KS test
    """

    pdf = df.select(predCol, labelCol).toPandas()
    rvs, cdf = pdf[predCol].values, pdf[labelCol].values

    return kstest(rvs, cdf).statistic


def kl_divergence(
    df : DataFrame,
    predCol : str,
    labelCol : str
) -> float:
    """
    Normalized Kullback–Leibler divergence on two dataframe columns. The normalization is
    as follows:

    .. math::
            \\frac{1}{1 + KL\_div}

    :param df: Dataframe with two target columns
    :param predCol: First column name
    :param labelCol: Second column name
    :return: Result of KL divergence
    """

    pdf = df.select(predCol, labelCol).toPandas()
    predicted, ground_truth = pdf[predCol].values, pdf[labelCol].values

    f_obs, edges = np.histogram(ground_truth)
    f_exp, _ = np.histogram(predicted, bins=edges)

    f_obs = f_obs.flatten() + 1e-5
    f_exp = f_exp.flatten() + 1e-5

    return 1 / (1 + np.sum(kl_div(f_obs, f_exp)))


# pylint: disable=too-few-public-methods
class EvaluateMetrics(ABC):
    """
    Recommendation systems and response function metric evaluator class.
    The class allows you to evaluate the quality of a response function on
    historical data or a recommender system on historical data or based on
    the results of an experiment in a simulator. Provides simultaneous
    calculation of several metrics using metrics from the Spark MLlib library.
    A created instance is callable on a dataframe with ``user_id, item_id,
    predicted relevance/response, true relevance/response`` format, which
    you can usually retrieve from simulators sample_responses() or log data
    with recommendation algorithm scores.
    """

    REGRESSION_METRICS = set(['rmse', 'mse', 'r2', 'mae', 'var'])
    MULTICLASS_METRICS = set([
        'f1', 'accuracy', 'weightedPrecision', 'weightedRecall',
        'weightedTruePositiveRate', 'weightedFalsePositiveRate',
        'weightedFMeasure', 'truePositiveRateByLabel', 'falsePositiveRateByLabel',
        'precisionByLabel', 'recallByLabel', 'fMeasureByLabel',
        'logLoss', 'hammingLoss'
    ])
    BINARY_METRICS = set(['areaUnderROC', 'areaUnderPR'])

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        userKeyCol : str,
        itemKeyCol : str,
        predictionCol : str,
        labelCol : str,
        mllib_metrics : Optional[Union[str, List[str]]] = None
    ):
        """
        :param userKeyCol: User identifier column name
        :param itemKeyCol: Item identifier column name
        :param predictionCol: Predicted scores column name
        :param labelCol: True label column name
        :param mllib_metrics: Metrics to calculate from spark's mllib. See
            REGRESSION_METRICS, MULTICLASS_METRICS, BINARY_METRICS for available
            values, defaults to None
        """

        super().__init__()

        self._userKeyCol = userKeyCol
        self._itemKeyCol = itemKeyCol
        self._predictionCol = predictionCol
        self._labelCol = labelCol

        if isinstance(mllib_metrics, str):
            mllib_metrics = [mllib_metrics]

        if mllib_metrics is None:
            mllib_metrics = []

        self._mllib_metrics = mllib_metrics

    def __call__(
        self,
        df : DataFrame
    ) -> Dict[str, float]:
        """
        Performs metrics calculations on passed dataframe

        :param df: Spark dataframe with userKeyCol, itemKeyCol, predictionCol
            and labelCol columns
        :return: Dictionary with metrics
        """

        df = df.withColumnRenamed(self._userKeyCol, 'user_idx')\
               .withColumnRenamed(self._itemKeyCol, 'item_idx')

        result = {}

        for m in self._mllib_metrics:
            evaluator = self._get_evaluator(m)
            result[m] = evaluator.evaluate(df)

        return result

    def _reg_or_multiclass_params(self):
        return {'predictionCol' : self._predictionCol, 'labelCol' : self._labelCol}

    def _binary_params(self):
        return {'rawPredictionCol' : self._predictionCol, 'labelCol' : self._labelCol}

    def _get_evaluator(self, metric):
        if metric in self.REGRESSION_METRICS:
            return RegressionEvaluator(
                metricName=metric, **self._reg_or_multiclass_params())
        if metric in self.BINARY_METRICS:
            return BinaryClassificationEvaluator(
                metricName=metric, **self._binary_params())
        if metric in self.MULTICLASS_METRICS:
            return MulticlassClassificationEvaluator(
                metricName=metric, **self._reg_or_multiclass_params())

        raise ValueError(f'Non existing metric was passed: {metric}')
