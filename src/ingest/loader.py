"""CSV loader for supermarket dataset."""

import pandas as pd
from pathlib import Path
from typing import Iterator, List, Dict, Any
from src.config import DATA_PATH


def load_csv() -> pd.DataFrame:
    """
    Load the supermarket dataset from CSV.
    
    Returns:
        pd.DataFrame: Raw dataset with all columns
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} products from {DATA_PATH}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """
    Validate that the CSV has the expected schema.
    
    Args:
        df: DataFrame to validate
        
    Raises:
        ValueError: If schema is invalid
    """
    expected_columns = {
        'product_id', 'sku', 'name', 'brand', 'variant', 'category',
        'origin', 'weight_kg', 'price_eur', 'in_stock', 'rating',
        'num_reviews', 'is_food', 'calories_per_100g', 'protein_g_per_100g',
        'sugar_g_per_100g', 'fat_g_per_100g', 'carbs_g_per_100g',
        'fiber_g_per_100g', 'sodium_mg_per_100g'
    }
    
    actual_columns = set(df.columns)
    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns
    
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if extra:
        print(f"Warning: Extra columns found: {extra}")
    
    print(f"Schema validation passed: {len(df)} rows, {len(actual_columns)} columns")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardize the data.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    # Make a copy to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert boolean columns
    df['in_stock'] = df['in_stock'].map({'Yes': True, 'No': False})
    df['is_food'] = df['is_food'].map({'Yes': True, 'No': False})
    
    # Fill NaN values for non-food items
    nutrition_cols = [
        'calories_per_100g', 'protein_g_per_100g', 'sugar_g_per_100g',
        'fat_g_per_100g', 'carbs_g_per_100g', 'fiber_g_per_100g',
        'sodium_mg_per_100g'
    ]
    for col in nutrition_cols:
        df[col] = df[col].fillna(0.0)
    
    # Ensure numeric types
    numeric_cols = [
        'product_id', 'weight_kg', 'price_eur', 'rating', 'num_reviews'
    ] + nutrition_cols
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle missing values in critical fields
    df['name'] = df['name'].fillna('Unknown Product')
    df['brand'] = df['brand'].fillna('Unknown Brand')
    df['category'] = df['category'].fillna('Unknown Category')
    
    print(f"Data cleaning completed. Final shape: {df.shape}")
    return df


def load_and_clean() -> pd.DataFrame:
    """
    Load, validate, and clean the dataset.
    
    Returns:
        pd.DataFrame: Cleaned dataset ready for processing
    """
    df = load_csv()
    validate_schema(df)
    df = clean_data(df)
    return df


def get_data_stats(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get basic statistics about the dataset.
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dict with statistics
    """
    stats = {
        "total_products": len(df),
        "food_products": df['is_food'].sum(),
        "non_food_products": (~df['is_food']).sum(),
        "in_stock": df['in_stock'].sum(),
        "out_of_stock": (~df['in_stock']).sum(),
        "categories": df['category'].nunique(),
        "brands": df['brand'].nunique(),
        "origins": df['origin'].nunique(),
        "avg_price": df['price_eur'].mean(),
        "price_range": {
            "min": df['price_eur'].min(),
            "max": df['price_eur'].max(),
            "median": df['price_eur'].median()
        }
    }
    return stats


if __name__ == "__main__":
    # Test the loader
    df = load_and_clean()
    stats = get_data_stats(df)
    print("\nDataset Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
