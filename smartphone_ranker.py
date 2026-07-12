import argparse
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED_COLUMNS = ['SMARTPHONENAME', 'BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM', 'PRICE']
NUMERIC_COLUMNS = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM', 'PRICE']


def validate_dataframe(df):
    """
    Validate and sanitize the smartphone ranking dataset.

    Args:
        df (pandas.DataFrame): Dataset containing the required smartphone
            feature columns and numeric scoring inputs.

    Returns:
        pandas.DataFrame: The validated dataframe, with missing numeric values
            imputed using column medians when necessary.

    Raises:
        ValueError: If a required column is missing or if any numeric feature
            column contains non-numeric values.

    Example:
        >>> clean_df = validate_dataframe(raw_df)
        >>> set(REQUIRED_COLUMNS).issubset(clean_df.columns)
        True
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
    for col in NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must contain numeric values")
    nulls = df[NUMERIC_COLUMNS].isnull().sum()
    if nulls.any():
        print(f"Warning: Found {nulls.sum()} null values - imputing with column medians")
        for col in NUMERIC_COLUMNS:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
    return df


class SmartphoneRanker:
    def __init__(self, data):
        """
        Initialize a ranking workflow with smartphone feature data.

        Args:
            data (pandas.DataFrame): Raw smartphone dataset containing the
                required feature and price columns.

        Returns:
            None: This initializer stores validated data and prepares empty
            placeholders for normalized values, weighted values, and scores.

        Raises:
            ValueError: If the supplied dataframe fails validation for required
                columns or numeric feature types.

        Example:
            >>> ranker = SmartphoneRanker(smartphone_df)
            >>> ranker.normalized_data is None
            True
        """
        self.data = validate_dataframe(data.copy())
        self.normalized_data = None
        self.weighted_data = None
        self.topsis_scores = None
        
    def normalize_data(self, features):
        """
        Normalize feature columns using TOPSIS vector normalization.

        Args:
            features (list[str]): Feature column names to normalize.

        Returns:
            pandas.DataFrame: Copy of the dataset with the selected feature
            columns normalized by their vector magnitude.

        Raises:
            KeyError: If any requested feature is not present in the dataset.
            ZeroDivisionError: If a feature column contains only zeros and
                cannot be vector-normalized.

        Example:
            >>> ranker = SmartphoneRanker(smartphone_df)
            >>> normalized = ranker.normalize_data(["BATTERY", "PRICE"])
            >>> normalized["BATTERY"].max() <= 1
            True
        """
        normalized = self.data.copy()
        
        for feature in features:
            # Vector normalization: x_ij / sqrt(sum(x_ij^2))
            sum_of_squares = np.sqrt(np.sum(self.data[feature]**2))
            normalized[feature] = self.data[feature] / sum_of_squares
            
        self.normalized_data = normalized
        return normalized
    
    def apply_weights(self, features, weights):
        """
        Apply decision weights to normalized feature columns.

        Args:
            features (list[str]): Feature column names to weight.
            weights (dict[str, float]): Mapping of feature names to their
                TOPSIS weights. Missing features default to a weight of 1.0.

        Returns:
            pandas.DataFrame: Copy of the normalized dataset with weighted
            feature values.

        Raises:
            ValueError: If normalization has not been run before weighting.
            KeyError: If any requested feature is not present in the normalized
                dataset.

        Example:
            >>> ranker.normalize_data(["BATTERY", "PRICE"])
            >>> weighted = ranker.apply_weights(["BATTERY", "PRICE"], {"BATTERY": 0.6, "PRICE": 0.4})
            >>> "BATTERY" in weighted
            True
        """
        if self.normalized_data is None:
            raise ValueError("Data must be normalized first")
            
        weighted = self.normalized_data.copy()
        
        for feature in features:
            weighted[feature] = self.normalized_data[feature] * weights.get(feature, 1.0)
            
        self.weighted_data = weighted
        return weighted
    
    def calculate_topsis(self, features, beneficial=['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR'], 
                         non_beneficial=['PRICE']):
        """
        Calculate TOPSIS scores and ranks for the weighted dataset.

        Args:
            features (list[str]): Feature column names included in the TOPSIS
                distance calculation.
            beneficial (list[str], optional): Features where higher values are
                preferred. Defaults to battery, camera, storage, and processor.
            non_beneficial (list[str], optional): Features where lower values
                are preferred. Defaults to price.

        Returns:
            pandas.DataFrame: Ranked smartphone dataset sorted by ascending
            rank, with `TOPSIS SCORE` and `RANK` columns added.

        Raises:
            ValueError: If weights have not been applied before scoring.
            KeyError: If any requested feature is missing from the weighted
                dataset.

        Example:
            >>> ranker.normalize_data(features)
            >>> ranker.apply_weights(features, weights)
            >>> result = ranker.calculate_topsis(features)
            >>> {"TOPSIS SCORE", "RANK"}.issubset(result.columns)
            True
        """
        if self.weighted_data is None:
            raise ValueError("Weights must be applied first")
        
        # Identify ideal best and ideal worst
        ideal_best = {}
        ideal_worst = {}
        
        for feature in features:
            if feature in beneficial:
                ideal_best[feature] = self.weighted_data[feature].max()
                ideal_worst[feature] = self.weighted_data[feature].min()
            else:
                ideal_best[feature] = self.weighted_data[feature].min()
                ideal_worst[feature] = self.weighted_data[feature].max()
        
        # Calculate Euclidean distance from ideal best and worst
        distance_to_best = []
        distance_to_worst = []
        
        for idx in self.weighted_data.index:
            dist_best = 0
            dist_worst = 0
            
            for feature in features:
                dist_best += (self.weighted_data.loc[idx, feature] - ideal_best[feature])**2
                dist_worst += (self.weighted_data.loc[idx, feature] - ideal_worst[feature])**2
            
            distance_to_best.append(np.sqrt(dist_best))
            distance_to_worst.append(np.sqrt(dist_worst))
        
        # Calculate TOPSIS score
        distance_to_best = np.array(distance_to_best)
        distance_to_worst = np.array(distance_to_worst)
        
        denominator = distance_to_best + distance_to_worst
        topsis_scores = np.divide(
            distance_to_worst,
            denominator,
            out=np.ones_like(distance_to_worst, dtype=float),
            where=denominator != 0
        )
        
        self.topsis_scores = topsis_scores
        
        # Add scores to original data
        result = self.data.copy()
        result['TOPSIS SCORE'] = np.round(topsis_scores, 3)
        result['RANK'] = result['TOPSIS SCORE'].rank(ascending=False, method='min').astype(int)
        
        return result.sort_values('RANK')
    
    def visualize_rankings(self, result_df):
        """
        Create and save ranking visualizations.

        Args:
            result_df (pandas.DataFrame): Ranked smartphone dataframe containing
                `SMARTPHONENAME`, feature columns, `TOPSIS SCORE`, `RANK`, and
                `PRICE`.

        Returns:
            None: Saves `smartphone_rankings.png` and displays the generated
            matplotlib figure.

        Raises:
            KeyError: If required ranking or feature columns are missing.
            ValueError: If feature normalization for the radar chart encounters
                invalid ranges.

        Example:
            >>> result = ranker.calculate_topsis(features)
            >>> ranker.visualize_rankings(result)
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Bar chart of TOPSIS scores
        ax1 = axes[0, 0]
        colors = plt.cm.viridis(np.linspace(0, 1, len(result_df)))
        ax1.barh(result_df['SMARTPHONENAME'], result_df['TOPSIS SCORE'], color=colors)
        ax1.set_xlabel('TOPSIS Score')
        ax1.set_title('Smartphone Rankings by TOPSIS Score')
        ax1.invert_yaxis()
        
        # 2. Feature comparison radar chart
        ax2 = axes[0, 1]
        features = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR']
        
        # Normalize features for radar chart
        normalized_features = result_df[features].copy()
        for feat in features:
            max_val = result_df[feat].max()
            min_val = result_df[feat].min()
            normalized_features[feat] = (result_df[feat] - min_val) / (max_val - min_val)
        
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
        angles += angles[:1]
        
        ax2 = plt.subplot(222, projection='polar')
        for idx, row in result_df.iterrows():
            values = normalized_features.loc[idx].values.tolist()
            values += values[:1]
            ax2.plot(angles, values, 'o-', linewidth=2, label=row['SMARTPHONENAME'])
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(features)
        ax2.set_ylim(0, 1)
        ax2.set_title('Feature Comparison (Normalized)')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax2.grid(True)
        
        # 3. Price vs TOPSIS Score scatter
        ax3 = axes[1, 0]
        scatter = ax3.scatter(result_df['PRICE'], result_df['TOPSIS SCORE'], 
                             c=result_df['RANK'], cmap='viridis_r', s=200, alpha=0.6)
        
        for idx, row in result_df.iterrows():
            ax3.annotate(row['SMARTPHONENAME'], 
                        (row['PRICE'], row['TOPSIS SCORE']),
                        fontsize=8, ha='center')
        
        ax3.set_xlabel('Price')
        ax3.set_ylabel('TOPSIS Score')
        ax3.set_title('Price vs Performance')
        plt.colorbar(scatter, ax=ax3, label='Rank')
        
        # 4. Feature heatmap
        ax4 = axes[1, 1]
        feature_data = result_df[['SMARTPHONENAME'] + features].set_index('SMARTPHONENAME')
        
        # Normalize for heatmap
        feature_data_norm = (feature_data - feature_data.min()) / (feature_data.max() - feature_data.min())
        
        sns.heatmap(feature_data_norm, annot=True, fmt='.2f', cmap='YlOrRd', 
                   ax=ax4, cbar_kws={'label': 'Normalized Value'})
        ax4.set_title('Feature Heatmap (Normalized)')
        
        plt.tight_layout()
        plt.savefig('smartphone_rankings.png', dpi=300, bbox_inches='tight')
        print("\nVisualization saved as 'smartphone_rankings.png'")
        plt.show()


def parse_args():
    """
    Parse command-line arguments for the ranking script.

    Args:
        None.

    Returns:
        argparse.Namespace: Parsed arguments including optional dataset path,
            result limit, export format, and feature weights.

    Raises:
        SystemExit: If invalid command-line arguments are provided.

    Example:
        >>> # From the shell:
        >>> # python smartphone_ranker.py --top 3 --export csv
    """
    parser = argparse.ArgumentParser(description="Smartphone Feature Ranking System")
    parser.add_argument("--data", type=str, help="Path to CSV dataset")
    parser.add_argument("--top", type=int, default=None, help="Show only top N results")
    parser.add_argument("--export", type=str, choices=["csv", "none"], default="csv", help="Export format")
    parser.add_argument("--weights", type=str, nargs="*", metavar="FEATURE=VALUE", help="Custom weights e.g. battery=0.25 camera=0.25")
    return parser.parse_args()


def main():
    """
    Run the end-to-end smartphone ranking command-line workflow.

    Args:
        None.

    Returns:
        None: Prints the ranking workflow, optionally exports CSV results, and
            generates ranking visualizations.

    Raises:
        ValueError: If the loaded or fallback dataset fails validation, if
            weight application is attempted before normalization, or if custom
            weights cannot be converted to numeric values.
        OSError: If output files cannot be written.

    Example:
        >>> # From the shell:
        >>> # python smartphone_ranker.py --data data/sample_smartphone_data.csv --top 5
    """
    args = parse_args()

    if args.data and os.path.exists(args.data):
        df = pd.read_csv(args.data)
        print(f"Loaded data from {args.data}")
    else:
        csv_path = os.path.join('data', 'sample_smartphone_data.csv')
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            print(f"Loaded data from {csv_path}")
        else:
            data = {
                'SMARTPHONENAME': ['Galaxy X1', 'Pixel pro', 'Moto one',
                                   'Redmi note', 'Realme GT'],
                'BATTERY':    [5000, 4500, 4000, 4500, 4200],
                'CAMERA':     [64,   48,   32,   50,   64],
                'STORAGE':    [128,  128,  64,   128,  256],
                'PROCESSOR':  [2.4,  2.8,  2.0,  2.3,  3.0],
                'RAM':        [8,    12,   6,    8,    12],
                'PRICE':      [20000, 25000, 18000, 15000, 30000]
            }
            df = pd.DataFrame(data)
            print("Using built-in sample dataset")

    print("=" * 80)
    print("SMARTPHONE FEATURE RANKING SYSTEM")
    print("=" * 80)

    print("\nOriginal Dataset:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    ranker = SmartphoneRanker(df)

    features = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM', 'PRICE']

    weights = {
        'BATTERY': 0.20, 'CAMERA': 0.20, 'STORAGE': 0.10,
        'PROCESSOR': 0.20, 'RAM': 0.15, 'PRICE': 0.15
    }
    if args.weights:
        for w in args.weights:
            if '=' in w:
                k, v = w.split('=', 1)
                key = k.upper().strip()
                if key in weights:
                    weights[key] = float(v)
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            weights = {k: v / total for k, v in weights.items()}

    print("\n" + "=" * 80)
    print("Feature Weights:")
    for feature, weight in weights.items():
        print(f"  {feature}: {weight * 100:.0f}%")
    print("=" * 80)

    print("\nStep 1: Normalizing data...")
    ranker.normalize_data(features)

    print("Step 2: Applying weights...")
    ranker.apply_weights(features, weights)

    print("Step 3: Calculating TOPSIS scores...")
    beneficial = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM']
    result = ranker.calculate_topsis(features, beneficial=beneficial,
                                     non_beneficial=['PRICE'])

    if args.top:
        result = result.head(args.top)

    print("\n" + "=" * 80)
    print("FINAL RANKINGS:")
    print("=" * 80)
    print(tabulate(result, headers='keys', tablefmt='grid', showindex=False))

    if args.export == "csv":
        output_path = os.path.join('output', 'final_smartphone_rankings.csv')
        os.makedirs('output', exist_ok=True)
        result.to_csv(output_path, index=False)
        print(f"\nResults saved to '{output_path}'")

    print("\nGenerating visualizations...")
    ranker.visualize_rankings(result)

    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    best_phone = result.iloc[0]
    print(f"\n  Best Overall : {best_phone['SMARTPHONENAME']}")
    print(f"  TOPSIS Score : {best_phone['TOPSIS SCORE']:.3f}")

    budget_phones = result[result['PRICE'] <= 20000]
    if not budget_phones.empty:
        best_budget = budget_phones.iloc[0]
        print(f"\n  Best Under 20k : {best_budget['SMARTPHONENAME']}")
        print(f"  Price          : Rs.{best_budget['PRICE']:,.0f}")
        print(f"  TOPSIS Score   : {best_budget['TOPSIS SCORE']:.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
