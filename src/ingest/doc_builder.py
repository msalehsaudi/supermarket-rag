"""Document builder for converting product rows to rich text."""

import pandas as pd
from langchain_core.documents import Document
from typing import List
from src.config import METADATA_FIELDS


def build_document_text(row: pd.Series) -> str:
    """
    Convert a product row to rich retrievable text.
    This text is what gets embedded — quality here = quality of retrieval.
    
    Args:
        row: Pandas Series representing a product
        
    Returns:
        Rich text description of the product
    """
    parts = [
        f"{row['name']} is a {row['category'].lower()} product",
        f"made by {row['brand']} from {row['origin']}.",
        f"It weighs {row['weight_kg']}kg and costs €{row['price_eur']}.",
    ]

    if row['is_food']:
        parts.append(
            f"Per 100g it contains {row['calories_per_100g']} calories, "
            f"{row['protein_g_per_100g']}g protein, "
            f"{row['fat_g_per_100g']}g fat, "
            f"{row['sugar_g_per_100g']}g sugar, "
            f"{row['carbs_g_per_100g']}g carbs, "
            f"{row['fiber_g_per_100g']}g fiber."
        )

    stock = "currently in stock" if row['in_stock'] else "currently out of stock"
    parts.append(f"This product is {stock} with a {row['rating']}/5 rating.")
    
    return " ".join(parts)


def build_metadata(row: pd.Series) -> dict:
    """
    Build metadata dictionary for a product.
    
    Args:
        row: Pandas Series representing a product
        
    Returns:
        Metadata dictionary with proper types
    """
    metadata = {}
    
    # Convert all fields to proper types according to schema
    for field, field_type in METADATA_FIELDS.items():
        value = row[field]
        
        if field_type == bool:
            metadata[field] = bool(value) if pd.notna(value) else False
        elif field_type == int:
            metadata[field] = int(value) if pd.notna(value) else 0
        elif field_type == float:
            metadata[field] = float(value) if pd.notna(value) else 0.0
        else:  # str
            metadata[field] = str(value) if pd.notna(value) else ""
    
    return metadata


def build_documents(df: pd.DataFrame) -> List[Document]:
    """
    Convert DataFrame to list of LangChain Documents.
    
    Args:
        df: Cleaned DataFrame of products
        
    Returns:
        List of Document objects with text and metadata
    """
    documents = []
    
    for idx, row in df.iterrows():
        text = build_document_text(row)
        metadata = build_metadata(row)
        
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"Built {len(documents)} documents from {len(df)} products")
    return documents


def filter_food_products(documents: List[Document]) -> List[Document]:
    """
    Filter documents to only include food products.
    
    Args:
        documents: List of all documents
        
    Returns:
        List of food product documents only
    """
    food_docs = [doc for doc in documents if doc.metadata.get('is_food', False)]
    print(f"Filtered to {len(food_docs)} food products from {len(documents)} total")
    return food_docs


def filter_in_stock(documents: List[Document]) -> List[Document]:
    """
    Filter documents to only include in-stock products.
    
    Args:
        documents: List of all documents
        
    Returns:
        List of in-stock product documents only
    """
    in_stock_docs = [doc for doc in documents if doc.metadata.get('in_stock', False)]
    print(f"Filtered to {len(in_stock_docs)} in-stock products from {len(documents)} total")
    return in_stock_docs


def get_document_stats(documents: List[Document]) -> dict:
    """
    Get statistics about the documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Statistics dictionary
    """
    stats = {
        "total_documents": len(documents),
        "food_products": sum(1 for doc in documents if doc.metadata.get('is_food', False)),
        "in_stock_products": sum(1 for doc in documents if doc.metadata.get('in_stock', False)),
        "categories": len(set(doc.metadata.get('category', '') for doc in documents)),
        "brands": len(set(doc.metadata.get('brand', '') for doc in documents)),
        "avg_text_length": sum(len(doc.page_content) for doc in documents) / len(documents) if documents else 0
    }
    return stats


if __name__ == "__main__":
    # Test the document builder
    from .loader import load_and_clean
    
    df = load_and_clean()
    documents = build_documents(df)
    
    print(f"\nSample document:")
    print(documents[0].page_content)
    print(f"\nSample metadata:")
    print(documents[0].metadata)
    
    stats = get_document_stats(documents)
    print(f"\nDocument statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
