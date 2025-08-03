import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import statsmodels.api as sm
import math
from scipy.stats import norm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    auc
)


class load_Gossis_data:
    """Loader and pre‑processing utilities for the public **GOSSIS** ICU dataset.

    The class is organised around two *stateless* helper functions that return
    fully processed copies of the original ``pandas`` DataFrames.  They are
    designed for exploratory workflows where console feedback (``print``)
    is desirable.

    Notes
    -----
    * No in‑place modification: every method operates on *copies* of the input
      frames, leaving originals intact.
    * Import prerequisites::

          import pandas as pd
          import numpy as np
          from sklearn.preprocessing import LabelEncoder
    """

    # ---------------------------------------------------------------------
    # 1. Basic loading / cleaning
    # ---------------------------------------------------------------------

    def load_gossis_data(labeled, unlabeled, target_feature="hospital_death",
                          sensitive_group="African American",
                          blackbox_feature="apache_4a_hospital_death_prob"):
        """Load, clean, and lightly transform the GOSSIS dataset.

        Steps performed
        --------------
        1. **Remove ID columns** – Columns whose name contains the substring
           ``"id"`` are dropped from both *labeled* and *unlabeled* sets.
        2. **Filter by missingness** – Columns with ≥10 % missing values in the
           labeled set are removed.
        3. **Row‑wise NA deletion** – Remaining rows containing *any* missing
           value are discarded.
        4. **Sensitive attribute encoding** – Column ``'ethnicity'`` is encoded
           as :math:`\{1,-1\}` where *1* denotes *sensitive_group*.
        5. **Black‑box scores** – Optionally extract predictions from a
           previously trained model (``blackbox_feature``).

        Parameters
        ----------
        labeled, unlabeled : pandas.DataFrame
            The GOSSIS dataset split into a *labeled* portion (contains the
            target) and an *unlabeled* portion.
        target_feature : str, default ``'hospital_death'``
            Name of the binary outcome column.
        sensitive_group : str, default ``'African American'``
            Ethnicity that defines the sensitive group.
        blackbox_feature : str | None, default
            Column containing pre‑computed probability scores. If ``None``, an
            empty Series is returned instead.

        Returns
        -------
        tuple
            *(labeled_clean, labeled_scores, labeled_sensitive,
            unlabeled_clean, unlabeled_scores, unlabeled_sensitive)*
        """
        print("=" * 60)
        print("STEP 1 – INITIAL LOAD & CLEANING")
        print("=" * 60)

        # Identify ID columns (heuristic: contains substring "id")
        id_cols = {col for col in labeled.columns if "id" in col}
        feature_cols = [col for col in labeled.columns if col not in id_cols]

        labeled_data = labeled[feature_cols].copy()
        unlabeled_data = unlabeled[feature_cols].copy()
        print(f"\nInitial shape: Labeled={labeled_data.shape}, Unlabeled={unlabeled_data.shape}")

        # Column‑wise NA filter (≥10 % threshold)
        nan_threshold = len(labeled_data) / 9  # ≈10 %
        features_to_keep = labeled_data.columns[labeled_data.isna().sum() < nan_threshold].tolist()
        print(f"\nFiltering columns (≤10 % NA) – keeping {len(features_to_keep)} features.")

        # Row‑wise NA deletion
        labeled_data = labeled_data[features_to_keep].dropna()
        unlabeled_data = unlabeled_data[list(set(features_to_keep) - {target_feature})].dropna()
        print(f"→ Shape after NA cleaning: Labeled={labeled_data.shape}, Unlabeled={unlabeled_data.shape}")

        # Binary encoding of the sensitive attribute
        print(f"\nEncoding sensitive group '{sensitive_group}' in column 'ethnicity' …")
        labeled_data["ethnicity"] = np.where(labeled_data["ethnicity"] == sensitive_group, 1, -1)
        unlabeled_data["ethnicity"] = np.where(unlabeled_data["ethnicity"] == sensitive_group, 1, -1)
        print("→ Unique values:", labeled_data["ethnicity"].unique())

        # Black‑box probability scores
        if blackbox_feature is None:
            print("\nNo black‑box score provided.")
            labeled_scores = pd.Series(name="blackbox_score")
            unlabeled_scores = pd.Series(name="blackbox_score")
        else:
            print(f"\nExtracting pre‑computed scores from '{blackbox_feature}' …")
            labeled_scores = labeled_data[blackbox_feature].copy()
            unlabeled_scores = unlabeled_data[blackbox_feature].copy()

        return (
            labeled_data, labeled_scores, labeled_data.ethnicity,
            unlabeled_data, unlabeled_scores, unlabeled_data.ethnicity,
        )

    # ---------------------------------------------------------------------
    # 2. Feature encoding
    # ---------------------------------------------------------------------

    def preprocess_features(
        labeled_data,
        unlabeled_data,
        target_feature,
        sensitive_feature,
        categorical_features,
        ordinal_features,
        no_change_features,
        do_scale=True,
    ):
        """Encode categorical and ordinal variables for downstream modelling.

        Workflow
        --------
        1. Concatenate labeled + unlabeled frames to ensure consistent column
           ordering after One‑Hot encoding.
        2. *Ordinal* variables → ``LabelEncoder``.
        3. *Categorical* variables → ``pd.get_dummies``.
        4. Re‑split the combined frame into processed labeled / unlabeled parts.

        Parameters
        ----------
        labeled_data, unlabeled_data : pandas.DataFrame
            Cleaned frames as returned by :pycode:`load_gossis_data`.
        target_feature : str
            Target column to be preserved only in the labeled set.
        sensitive_feature : str
            Column holding the binary sensitive attribute (already encoded).
        categorical_features, ordinal_features, no_change_features : list[str]
            Lists of column names grouped by encoding strategy.
        do_scale : bool, default ``True``
            Whether min‑max / z‑score scaling will be applied later. The function
            only prints a reminder; actual scaling happens downstream to avoid
            data leakage.

        Returns
        -------
        tuple
            *(labeled_processed, unlabeled_processed, labeled_original_snapshot)*
        """
        print("\n" + "=" * 60)
        print("STEP 2 – FEATURE ENGINEERING")
        print("=" * 60)

        all_features = (
            categorical_features + ordinal_features + no_change_features + [sensitive_feature]
        )
        labeled_data = labeled_data[all_features + [target_feature]]
        unlabeled_data = unlabeled_data[all_features]

        data_orig_names = labeled_data.copy()
        print("\nOriginal feature sample:")
        print(data_orig_names.head(3))
        print("Shape:", data_orig_names.shape)

        # Combine to guarantee identical encoding across sets
        print("\nConcatenating labeled + unlabeled for consistent encoding …")
        combined_data = pd.concat(
            [labeled_data.drop(columns=[target_feature]), unlabeled_data],
            ignore_index=True,
        )

        # Ordinal encoding
        print(f"\nLabel‑encoding {len(ordinal_features)} ordinal variable(s) …")
        for var in ordinal_features:
            combined_data[var] = LabelEncoder().fit_transform(combined_data[var])

        # One‑Hot encoding for categorical features
        print(f"\nOne‑Hot‑encoding {len(categorical_features)} categorical variable(s) …")
        shape_before = combined_data.shape
        combined_data = pd.get_dummies(combined_data, columns=categorical_features, dummy_na=False)
        print(f"→ Shape {shape_before} → {combined_data.shape}")

        # Split back
        labeled_processed = combined_data.iloc[: len(labeled_data)].copy()
        unlabeled_processed = combined_data.iloc[len(labeled_data) :].copy()
        labeled_processed[target_feature] = labeled_data[target_feature].values

        print("\nEncoded feature sample:")
        print(labeled_processed.head(3))

        if do_scale:
            print("\nNOTE – feature scaling left to downstream pipeline to prevent leakage.")

        return labeled_processed, unlabeled_processed, data_orig_names
    

    # ------------------------------------------------------------------
    # 3. Variable‑analysis utilities
    # ------------------------------------------------------------------

    def identify_variable_types(
        df,
        columns: list[str] | None = None,
        discrete_threshold: int = 10,
        unique_ratio_threshold: float = 0.05,
    ):
        """Classify columns as *continuous*, *discrete*, *binary*, or *categorical*.

        The heuristic relies on the number of distinct values (:pycode:`n_unique`)
        and the proportion of unique entries (:pycode:`unique_ratio`). Non‑numeric
        variables are labelled *categorical* outright.

        Parameters
        ----------
        df : pandas.DataFrame
            Input frame.
        columns : list[str] | None, default ``None``
            Subset of columns to analyse; if ``None`` all columns are processed.
        discrete_threshold : int, default ``10``
            Maximum number of unique values to consider a numeric variable as
            discrete.
        unique_ratio_threshold : float, default ``0.05``
            Minimum (unique / total) ratio for a variable to be regarded as
            *continuous* despite being integer‑valued.

        Returns
        -------
        tuple
            *(result_dict, summary_frame)* where ``result_dict`` contains the
            keys ``'continuous'``, ``'discrete'``, ``'categorical'``,
            ``'binary'`` and ``'summary'``. The ``summary_frame`` is a tidy
            :pycode:`DataFrame` with one row per analysed variable and the extra
            ``classification`` column.
        """
        if columns is None:
            columns = df.columns

        results = {
            "continuous": [],
            "discrete": [],
            "categorical": [],
            "binary": [],
            "summary": [],
        }

        for col in columns:
            # Non‑numeric ⇒ categorical
            if not pd.api.types.is_numeric_dtype(df[col]):
                results["categorical"].append(col)
                continue

            n_unique = df[col].nunique()
            n_total = len(df[col])
            unique_ratio = n_unique / n_total

            # Basic statistics snapshot
            summary = {
                "column": col,
                "dtype": df[col].dtype,
                "n_unique": n_unique,
                "unique_ratio": unique_ratio,
                "min": df[col].min(),
                "max": df[col].max(),
                "has_decimals": any(
                    x % 1 != 0 for x in df[col].dropna().sample(min(1000, len(df[col]))).values
                ),
            }
            results["summary"].append(summary)

            # Classification rules
            if n_unique == 2:
                results["binary"].append(col)
                results["discrete"].append(col)
            elif n_unique <= discrete_threshold or unique_ratio < unique_ratio_threshold:
                results["discrete"].append(col)
            else:
                # Check for decimals → continuous
                if summary["has_decimals"]:
                    results["continuous"].append(col)
                else:
                    # Many unique integers may still be identifiers
                    if n_unique > 100:
                        results["continuous"].append(col)
                    else:
                        results["discrete"].append(col)

        # Build tidy summary frame
        summary_df = pd.DataFrame(results["summary"])
        if not summary_df.empty:
            summary_df["classification"] = summary_df["column"].apply(
                lambda x: "binary"
                if x in results["binary"]
                else "discrete"
                if x in results["discrete"]
                else "continuous"
                if x in results["continuous"]
                else "categorical"
            )

        return results, summary_df

    def select_uncorrelated_features(df, threshold: float = 0.8):
        """Greedy filter that removes features with pairwise correlation > *threshold*.

        The algorithm retains the first occurrence of any highly correlated pair
        and drops the remaining columns. Correlations are computed on absolute
        values using ``df.corr().abs()``.

        Parameters
        ----------
        df : pandas.DataFrame
            Frame containing *only numeric predictors*.
        threshold : float, default ``0.8``
            Correlation magnitude above which two features are considered
            redundant.

        Returns
        -------
        list[str]
            Names of the columns to **keep** (i.e. low pairwise correlation).
        """
        # Correlation matrix
        corr_matrix = df.corr().abs()

        # Upper‑triangle mask
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Columns to drop – any with correlation above the threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

        # Columns to keep
        to_keep = [column for column in df.columns if column not in to_drop]

        return to_keep

