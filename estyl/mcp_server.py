# server.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import sys
from mcp.server.fastmcp import FastMCP
from .core import retrieve, SingleSearchParams, OutfitParams
from .config import CATEGORY_OPTIONS, BUDGET_TIERS, CAT_SYNONYMS

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='mcp_server.log', 
    filemode='w'
)

logger = logging.getLogger("mcp_server")

mcp = FastMCP("Estyl Retriever")

CATEGORIES = {
    "Tops",
    "Bottoms",
    "Shoes",
    "Outerwear",
    "Jewelry & Watches",
    "Accessories",
    "Activewear",
    "Suits & Tailoring",
    "Underwear",
    "Bags",
    "Dresses & One-Pieces",
    "Lingerie/Underwear",
    "Small Leather Goods",
    "Swimwear",
}

# Map synonyms / user words → canonical categories
CAT_SYNONYMS = {
    "shirt": "Tops",
    "tshirt": "Tops",
    "tee": "Tops",
    "blouse": "Tops",
    "pants": "Bottoms",
    "jeans": "Bottoms",
    "shorts": "Bottoms",
    "skirt": "Bottoms",
    "sneakers": "Shoes",
    "heels": "Shoes",
    "boots": "Shoes",
    "sandals": "Shoes",
    "jacket": "Outerwear",
    "coat": "Outerwear",
    "bracelet": "Jewelry & Watches",
    "earring": "Jewelry & Watches",
    "ring": "Jewelry & Watches",
    "watch": "Jewelry & Watches",
    "necklace": "Jewelry & Watches",
    "belt": "Accessories",
    "scarf": "Accessories",
    "hat": "Accessories",
    "cap": "Accessories",
    "bag": "Bags",
    "purse": "Bags",
    "handbag": "Bags",
    "backpack": "Bags",
    "dress": "Dresses & One-Pieces",
    "swimsuit": "Swimwear",
    "bikini": "Swimwear",
    "lingerie": "Lingerie/Underwear",
    "underwear": "Underwear",
    "wallet": "Small Leather Goods",
}

def normalize_categories(user_query: str) -> List[str]:
    """Extract categories from user query using CAT_SYNONYMS."""
    q = user_query.lower()
    found = set()
    for word, cat in CAT_SYNONYMS.items():
        if word in q:
            found.add(cat)
    return list(found & CATEGORIES)

@mcp.tool(title="Retrieve fashion items or compose outfits", name="estyl_retrieve")
def estyl_retrieve(
    
    mode: str,
    # Common query hints
    text_query: str, # e.g. "wedding", "casual party", "office formal"
    gender: str,  # "male" | "female" | "unisex"
    categories: List[str],  
    search_with: str = "Text",  # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None,
    brand_contains: Optional[str] = None,
    budget_tier: str = "Mid",  # Budget buckets (approx): Budget <100, Mid 100-300, Premium 300-600, Luxury >600
    budget: Optional[float] = None,  # Explicit budget in euros/dollars overrides budget_tier
    # Single-item controls
    limit: int = 10,
    topk_for_rerank: int = 10,
    offset: int = 0,
    exclude_ids: Optional[List[str]] = None,
    # Outfit controls
    num_outfits: Optional[int] = 5,
    articles: Optional[int] = 5,  # 2..5
    per_cat_candidates: Optional[int] = 5,
) -> Dict[str, Any]:
    """
    mode: "single" or "outfit"
    text_query: Global theme - Short description of the event or style (e.g. "wedding", "beach vacation").
    gender: {"male","female","unisex"}.
    budget: Numeric budget cap. Example: "under 250 euros" -> budget=250.
    budget_tier: Only used if user does not specify numeric budget.
    categories: Will be auto-normalized to allowed taxonomy.

    SINGLE mode relevant args:
      - text_query, search_with, image_b64, gender, brand_contains, budget_tier, budget
      - categories, limit, topk_for_rerank, offset, exclude_ids

    OUTFIT mode relevant args:
      - text_query, search_with, image_b64, gender, brand_contains, budget_tier, budget
      - num_outfits, articles (2..5), per_cat_candidates
    """
    try:
        # Normalize categories
        logger.debug(">>> estyl_retrieve called with args: %s", locals())
        resolved_cats = normalize_categories(text_query)
        if categories:
            # filter user-passed categories too
            resolved_cats.extend([c for c in categories if c in CATEGORIES])
        resolved_cats = list(set(resolved_cats))

        logger.debug("Resolved categories: %s", resolved_cats)

        mode_l = (mode or "").lower()

        if mode_l.startswith("single"):
            s = SingleSearchParams(
                text_query=text_query,
                search_with=search_with,
                image_b64=image_b64,
                gender=gender,
                categories=resolved_cats,
                brand_contains=brand_contains,
                budget_tier=budget_tier,
                budget=budget if budget is not None else 350.0,
                limit=limit,
                topk_for_rerank=topk_for_rerank,
                offset=offset,
                exclude_ids=exclude_ids,
            )
            logger.debug("Estyl Single params: %s", s)
            return retrieve("single", single=s)

        elif mode_l.startswith("outfit"):
            o = OutfitParams(
                text_query=text_query,
                search_with=search_with,
                image_b64=image_b64,
                gender=gender,
                brand_contains=brand_contains,
                budget_tier=budget_tier,
                budget=budget if budget is not None else 350.0,
                num_outfits=num_outfits or 3,
                articles=articles or 3,
                per_cat_candidates=per_cat_candidates or 5,
            )
            logger.debug("Estyl Outfit params: %s", o)
            return retrieve("outfit", outfit=o)

        else:
            raise ValueError("mode must be 'single' or 'outfit'")

    except Exception as e:
        logger.exception("Retrieve failed")
        return {"error": str(e)}


@mcp.tool(title="Get supported options", name="estyl_options")
def estyl_options() -> Dict[str, Any]:
    """
    Introspection for UI/host: categories and available budget tiers per category.
    """
    return {
        "categories": CATEGORY_OPTIONS,
        "budget_tiers": {k: list(v.keys()) for k, v in BUDGET_TIERS.items()},
    }


def main():
    # Default transport is stdio for local/desktop hosts; can be 'streamable-http'
    mcp.run()


if __name__ == "__main__":
    main()