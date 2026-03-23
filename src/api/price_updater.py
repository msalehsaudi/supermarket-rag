"""Price update functionality for supermarket products."""

import pandas as pd
import asyncio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

from src.config import DATA_PATH, CHROMA_DB_PATH
from src.vectorstore.chroma_store import get_collection
from src.ingest.doc_builder import build_document_text, build_metadata
from src.ingest.embedder import embed_batch, AsyncOpenAI, OPENAI_API_KEY, EMBEDDING_MODEL


class PriceUpdater:
    """
    Handles price updates for supermarket products.
    Updates both the CSV file and vector store.
    """
    
    def __init__(self):
        """Initialize the price updater."""
        self.backup_dir = DATA_PATH.parent / "backups"
        self.backup_dir.mkdir(exist_ok=True)
    
    def _backup_csv(self) -> Path:
        """
        Create a backup of the current CSV file.
        
        Returns:
            Path to the backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"supermarket_dataset_backup_{timestamp}.csv"
        
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df.to_csv(backup_path, index=False)
            print(f"✅ CSV backed up to {backup_path}")
        else:
            raise FileNotFoundError(f"Original CSV not found at {DATA_PATH}")
        
        return backup_path
    
    def _update_csv_prices(
        self, 
        price_updates: Dict[int, float]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Update prices in the CSV file.
        
        Args:
            price_updates: Dictionary of product_id -> new_price
            
        Returns:
            Tuple of (updated_count, failed_updates)
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
        
        # Load current data
        df = pd.read_csv(DATA_PATH)
        
        updated_count = 0
        failed_updates = []
        old_prices = []
        
        for product_id, new_price in price_updates.items():
            try:
                # Find the product
                mask = df['product_id'] == product_id
                if not mask.any():
                    failed_updates.append({
                        "product_id": product_id,
                        "error": "Product not found"
                    })
                    continue
                
                # Get old price
                old_price = df.loc[mask, 'price_eur'].iloc[0]
                
                # Update price
                df.loc[mask, 'price_eur'] = new_price
                
                old_prices.append({
                    "product_id": product_id,
                    "old_price": float(old_price),
                    "new_price": new_price
                })
                
                updated_count += 1
                
            except Exception as e:
                failed_updates.append({
                    "product_id": product_id,
                    "error": str(e)
                })
        
        # Save updated CSV
        if updated_count > 0:
            df.to_csv(DATA_PATH, index=False)
            print(f"✅ Updated {updated_count} prices in CSV")
        
        return updated_count, failed_updates, old_prices
    
    async def _update_vector_store(
        self, 
        price_updates: Dict[int, float]
    ) -> Tuple[int, List[Dict[str, Any]]]:
        """
        Update prices in the vector store.
        
        Args:
            price_updates: Dictionary of product_id -> new_price
            
        Returns:
            Tuple of (updated_count, failed_updates)
        """
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required for vector store updates")
        
        collection = get_collection()
        client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        
        updated_count = 0
        failed_updates = []
        
        for product_id, new_price in price_updates.items():
            try:
                # Get existing document
                result = collection.get(
                    ids=[str(product_id)],
                    include=["metadatas", "documents"]
                )
                
                if not result['ids']:
                    failed_updates.append({
                        "product_id": product_id,
                        "error": "Product not found in vector store"
                    })
                    continue
                
                # Update metadata
                metadata = result['metadatas'][0].copy()
                metadata['price_eur'] = new_price
                
                # Rebuild document text with updated price
                # Convert metadata back to Series for document building
                import pandas as pd
                row_series = pd.Series(metadata)
                updated_text = build_document_text(row_series)
                
                # Re-embed the updated document
                embeddings = await embed_batch([updated_text], client)
                
                # Update in vector store
                collection.update(
                    ids=[str(product_id)],
                    embeddings=embeddings,
                    documents=[updated_text],
                    metadatas=[metadata]
                )
                
                updated_count += 1
                
            except Exception as e:
                failed_updates.append({
                    "product_id": product_id,
                    "error": str(e)
                })
        
        if updated_count > 0:
            print(f"✅ Updated {updated_count} prices in vector store")
        
        return updated_count, failed_updates
    
    async def update_prices(
        self, 
        product_ids: List[int], 
        new_prices: List[float],
        reason: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Update prices for specified products.
        
        Args:
            product_ids: List of product IDs to update
            new_prices: List of new prices
            reason: Optional reason for the update
            
        Returns:
            Dictionary with update results
        """
        if len(product_ids) != len(new_prices):
            raise ValueError("product_ids and new_prices must have the same length")
        
        if len(product_ids) > 100:
            raise ValueError("Cannot update more than 100 products at once")
        
        # Validate prices
        for price in new_prices:
            if price < 0:
                raise ValueError("Prices cannot be negative")
        
        # Create price update dictionary
        price_updates = dict(zip(product_ids, new_prices))
        
        # Create backup
        backup_path = self._backup_csv()
        
        # Update CSV
        csv_updated, csv_failed, old_prices = self._update_csv_prices(price_updates)
        
        # Update vector store
        vs_updated, vs_failed = await self._update_vector_store(price_updates)
        
        # Combine failed updates
        all_failed = []
        for failed in csv_failed + vs_failed:
            if failed not in all_failed:
                all_failed.append(failed)
        
        # Log the update
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            "timestamp": timestamp,
            "reason": reason or "Manual update",
            "backup_file": str(backup_path),
            "products_updated": csv_updated,
            "csv_updates": csv_updated,
            "vector_store_updates": vs_updated,
            "failed_updates": len(all_failed)
        }
        
        print(f"📊 Price update completed: {log_entry}")
        
        return {
            "updated_products": csv_updated,
            "failed_updates": all_failed,
            "old_prices": old_prices,
            "backup_file": str(backup_path),
            "timestamp": timestamp,
            "reason": reason
        }
    
    def get_price_history(self, product_id: int) -> List[Dict[str, any]]:
        """
        Get price history for a product (placeholder implementation).
        
        Args:
            product_id: Product ID to get history for
            
        Returns:
            List of price history entries
        """
        # This would typically read from a price history log
        # For now, return a placeholder
        return [
            {
                "date": "2024-01-01",
                "price": 10.99,
                "reason": "Initial price"
            }
        ]
    
    def bulk_price_adjustment(
        self, 
        adjustment_type: str,
        adjustment_value: float,
        category_filter: Optional[str] = None,
        brand_filter: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Apply bulk price adjustments.
        
        Args:
            adjustment_type: Type of adjustment ('percentage', 'absolute')
            adjustment_value: Value for adjustment
            category_filter: Optional category filter
            brand_filter: Optional brand filter
            
        Returns:
            Dictionary with adjustment results
        """
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"CSV file not found at {DATA_PATH}")
        
        df = pd.read_csv(DATA_PATH)
        
        # Apply filters
        mask = pd.Series([True] * len(df))
        
        if category_filter:
            mask &= df['category'] == category_filter
        
        if brand_filter:
            mask &= df['brand'] == brand_filter
        
        filtered_df = df[mask]
        
        if filtered_df.empty:
            return {
                "affected_products": 0,
                "message": "No products match the specified filters"
            }
        
        # Calculate new prices
        if adjustment_type == "percentage":
            factor = 1 + (adjustment_value / 100)
            filtered_df.loc[:, 'new_price'] = filtered_df['price_eur'] * factor
        elif adjustment_type == "absolute":
            filtered_df.loc[:, 'new_price'] = filtered_df['price_eur'] + adjustment_value
        else:
            raise ValueError("adjustment_type must be 'percentage' or 'absolute'")
        
        # Ensure no negative prices
        filtered_df.loc[filtered_df['new_price'] < 0, 'new_price'] = 0.01
        
        # Create update lists
        product_ids = filtered_df['product_id'].tolist()
        new_prices = filtered_df['new_price'].tolist()
        
        reason = f"Bulk {adjustment_type} adjustment of {adjustment_value}"
        if category_filter:
            reason += f" for category {category_filter}"
        if brand_filter:
            reason += f" for brand {brand_filter}"
        
        print(f"🔄 Applying bulk adjustment to {len(product_ids)} products")
        
        # This would need to be run in an async context
        # For now, return the preparation results
        return {
            "affected_products": len(product_ids),
            "product_ids": product_ids,
            "new_prices": new_prices,
            "reason": reason,
            "message": f"Prepared {len(product_ids)} products for price adjustment"
        }


# Convenience function for API usage
async def update_product_prices(
    product_ids: List[int], 
    new_prices: List[float],
    reason: Optional[str] = None
) -> Dict[str, any]:
    """
    Update prices for products (convenience function for API).
    
    Args:
        product_ids: List of product IDs
        new_prices: List of new prices
        reason: Optional reason
        
    Returns:
        Update results
    """
    updater = PriceUpdater()
    return await updater.update_prices(product_ids, new_prices, reason)


if __name__ == "__main__":
    # Test the price updater
    async def test():
        updater = PriceUpdater()
        
        # Test a small price update
        result = await updater.update_prices(
            product_ids=[1, 2, 3],
            new_prices=[1.99, 2.99, 3.99],
            reason="Test update"
        )
        
        print("Price update test result:")
        print(result)
    
    asyncio.run(test())
