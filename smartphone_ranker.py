import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

class SmartphoneRanker:
    def __init__(self, data):
        """
        Initialize the Smartphone Ranker with dataset
        
        Parameters:
        data: DataFrame with smartphone features
        """
        self.data = data.copy()
        self.normalized_data = None
        self.weighted_data = None
        self.topsis_scores = None
        
    def normalize_data(self, features):
        """
        Normalize the feature data using vector normalization
        
        Parameters:
        features: list of feature column names to normalize
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
        Apply weights to normalized features
        
        Parameters:
        features: list of feature column names
        weights: dictionary with feature names as keys and weights as values
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
        Calculate TOPSIS scores
        
        Parameters:
        features: list of all feature column names
        beneficial: list of features where higher is better
        non_beneficial: list of features where lower is better
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
        
        topsis_scores = distance_to_worst / (distance_to_best + distance_to_worst)
        
        self.topsis_scores = topsis_scores
        
        # Add scores to original data
        result = self.data.copy()
        result['TOPSIS SCORE'] = np.round(topsis_scores, 3)
        result['RANK'] = result['TOPSIS SCORE'].rank(ascending=False, method='min').astype(int)
        
        return result.sort_values('RANK')
    
    def visualize_rankings(self, result_df):
        """
        Create visualizations for the rankings
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


def main():
    """
    Main function to run the smartphone ranking system.
    Supports both sample dataset and loading from CSV file.
    """
    import os

    # ----------------------------------------------------------------
    # Option 1: Load from CSV (recommended — uses your full dataset)
    # ----------------------------------------------------------------
    csv_path = os.path.join('data', 'sample_smartphone_data.csv')
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Loaded data from {csv_path}")
    else:
        # ----------------------------------------------------------------
        # Option 2: Built-in sample dataset (CORRECTED — PROCESSOR fixed)
        # PROCESSOR values are clock speeds in GHz, NOT price values.
        # ----------------------------------------------------------------
        data = {
            'SMARTPHONENAME': ['Galaxy X1', 'Pixel pro', 'Moto one',
                               'Redmi note', 'Realme GT'],
            'BATTERY':    [5000, 4500, 4000, 4500, 4200],   # mAh
            'CAMERA':     [64,   48,   32,   50,   64],      # MP
            'STORAGE':    [128,  128,  64,   128,  256],     # GB
            'PROCESSOR':  [2.4,  2.8,  2.0,  2.3,  3.0],   # GHz  ← FIXED
            'RAM':        [8,    12,   6,    8,    12],      # GB
            'PRICE':      [20000, 25000, 18000, 15000, 30000]  # INR
        }
        df = pd.DataFrame(data)
        print("Using built-in sample dataset")

    print("=" * 80)
    print("SMARTPHONE FEATURE RANKING SYSTEM")
    print("=" * 80)

    print("\nOriginal Dataset:")
    print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))

    # Initialize ranker
    ranker = SmartphoneRanker(df)

    # Define features to analyse (excluding SMARTPHONENAME)
    features = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM', 'PRICE']

    # Define weights for each feature (must sum to 1.0)
    weights = {
        'BATTERY':   0.20,   # 20% — battery life matters
        'CAMERA':    0.20,   # 20% — camera quality
        'STORAGE':   0.10,   # 10% — storage space
        'PROCESSOR': 0.20,   # 20% — processing speed
        'RAM':       0.15,   # 15% — multitasking
        'PRICE':     0.15    # 15% — lower price is better
    }

    print("\n" + "=" * 80)
    print("Feature Weights:")
    for feature, weight in weights.items():
        print(f"  {feature}: {weight * 100:.0f}%")
    print("=" * 80)

    # Step 1: Normalize data
    print("\nStep 1: Normalizing data...")
    ranker.normalize_data(features)

    # Step 2: Apply weights
    print("Step 2: Applying weights...")
    ranker.apply_weights(features, weights)

    # Step 3: Calculate TOPSIS scores
    print("Step 3: Calculating TOPSIS scores...")
    beneficial = ['BATTERY', 'CAMERA', 'STORAGE', 'PROCESSOR', 'RAM']
    result = ranker.calculate_topsis(features, beneficial=beneficial,
                                     non_beneficial=['PRICE'])

    # Display results
    print("\n" + "=" * 80)
    print("FINAL RANKINGS:")
    print("=" * 80)
    print(tabulate(result, headers='keys', tablefmt='grid', showindex=False))

    # Save to CSV
    output_path = os.path.join('output', 'final_smartphone_rankings.csv')
    os.makedirs('output', exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"\nResults saved to '{output_path}'")

    # Generate visualizations
    print("\nGenerating visualizations...")
    ranker.visualize_rankings(result)

    # Additional insights
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
        print(f"  Price          : ₹{best_budget['PRICE']:,.0f}")
        print(f"  TOPSIS Score   : {best_budget['TOPSIS SCORE']:.3f}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()