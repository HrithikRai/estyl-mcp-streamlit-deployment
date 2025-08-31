# server.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from .core import retrieve, SingleSearchParams, OutfitParams
from .config import CATEGORY_OPTIONS, BUDGET_TIERS

logging.basicConfig(level=logging.DEBUG)
mcp = FastMCP("Estyl Retriever")


@mcp.tool(title="Retrieve fashion items or compose outfits", name="estyl_retrieve")
def estyl_retrieve(
    mode: str,
    # Common query hints
    text_query: Optional[str] = None,
    search_with: str = "Text",  # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None,
    gender: str = "any",  # "any" | "male" | "female" | "unisex"
    brand_contains: Optional[str] = None,
    budget_tier: str = "Mid",  # "Budget" | "Mid" | "Premium" | "Luxury"
    budget: Optional[float] = None,  # overall cap; defaults to 350.0 if None
    # Single-item controls
    categories: Optional[List[str]] = None,
    limit: int = 12,
    topk_for_rerank: int = 40,
    offset: int = 0,
    exclude_ids: Optional[List[str]] = None,
    # Outfit controls
    num_outfits: Optional[int] = None,
    articles: Optional[int] = None,  # 2..5
    per_cat_candidates: Optional[int] = None,
) -> Dict[str, Any]:
    """
    mode: "single" or "outfit"

    SINGLE mode relevant args:
      - text_query, search_with, image_b64, gender, brand_contains, budget_tier, budget
      - categories, limit, topk_for_rerank, offset, exclude_ids

    OUTFIT mode relevant args:
      - text_query, search_with, image_b64, gender, brand_contains, budget_tier, budget
      - num_outfits, articles (2..5), per_cat_candidates
    """
    try:
        mode_l = (mode or "").lower()
        if mode_l.startswith("single"):
            s = SingleSearchParams(
                text_query=text_query or "",
                search_with=search_with,
                image_b64=image_b64,
                gender=gender,
                categories=(categories or []),
                brand_contains=brand_contains,
                budget_tier=budget_tier,
                budget=budget if budget is not None else 350.0,
                limit=limit,
                topk_for_rerank=topk_for_rerank,
                offset=offset,
                exclude_ids=exclude_ids,
            )
            logging.debug("Estyl Single params: %s", s)
            return retrieve("single", single=s)

        elif mode_l.startswith("outfit"):
            o = OutfitParams(
                text_query=text_query or "",
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
            logging.debug("Estyl Outfit params: %s", o)
            return retrieve("outfit", outfit=o)

        else:
            raise ValueError("mode must be 'single' or 'outfit'")

    except Exception as e:
        logging.exception("Retrieve failed")
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
