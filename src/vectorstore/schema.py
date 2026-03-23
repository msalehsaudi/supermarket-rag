"""Metadata schema definitions for the vector store."""

from typing import Dict, Type

# Metadata field definitions with their types
METADATA_FIELDS: Dict[str, Type] = {
    "product_id": int,
    "sku": str,
    "name": str,
    "brand": str,
    "category": str,
    "origin": str,
    "weight_kg": float,
    "price_eur": float,
    "in_stock": bool,
    "rating": float,
    "is_food": bool,
    "calories_per_100g": float,
    "protein_g_per_100g": float,
    "fat_g_per_100g": float,
    "sugar_g_per_100g": float,
    "carbs_g_per_100g": float,
    "fiber_g_per_100g": float,
    "sodium_mg_per_100g": float,
}

# Fields that are searchable via metadata filters
FILTERABLE_FIELDS = [
    "price_eur",
    "calories_per_100g", 
    "protein_g_per_100g",
    "fat_g_per_100g",
    "sugar_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sodium_mg_per_100g",
    "weight_kg",
    "rating",
    "category",
    "brand",
    "origin",
    "is_food",
    "in_stock"
]

# Numeric fields that support range queries
NUMERIC_FIELDS = [
    "price_eur",
    "calories_per_100g",
    "protein_g_per_100g", 
    "fat_g_per_100g",
    "sugar_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sodium_mg_per_100g",
    "weight_kg",
    "rating"
]

# Categorical fields that support exact match
CATEGORICAL_FIELDS = [
    "category",
    "brand", 
    "origin",
    "is_food",
    "in_stock"
]

# Fields to display in product cards
DISPLAY_FIELDS = [
    "name",
    "brand",
    "category",
    "price_eur",
    "weight_kg",
    "rating",
    "in_stock"
]

# Nutrition fields for food products
NUTRITION_FIELDS = [
    "calories_per_100g",
    "protein_g_per_100g",
    "fat_g_per_100g", 
    "sugar_g_per_100g",
    "carbs_g_per_100g",
    "fiber_g_per_100g",
    "sodium_mg_per_100g"
]
