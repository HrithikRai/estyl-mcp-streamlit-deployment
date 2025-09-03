from __future__ import annotations
import io, json, time, itertools, base64
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional, List, Dict, Tuple, Any
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery
import heapq, math
from .config import (
    WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION,
    QUERY_PROPS, RETURN_PROPS, CATEGORY_OPTIONS,
    OUTFIT_WEIGHTS, OUTFIT_ORDER_BY_N, CAT_SYNONYMS,
)

import logging, sys

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filename='core.log', 
    filemode='w'
)

logger = logging.getLogger("core")

ALLOW_FCLIP_RERANK = os.getenv("ALLOW_FCLIP_RERANK", "True").lower() in ("1", "true", "yes")


# ------------------------
# Dataclasses / Types
# ------------------------
@dataclass
class SingleSearchParams:
    text_query: str = ""
    search_with: str = "Text"          # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None    # base64 encoded image
    gender: str = "unisex"                # "male" | "female" | "unisex"
    categories: List[str] = None
    brand_contains: Optional[str] = None
    budget: float = 350.0
    limit: int = 10
    topk_for_rerank: int = 5
    exclude_ids: Optional[List[str]] = None

@dataclass
class OutfitParams:
    text_query: str = ""               # event / vibe / style hints
    search_with: str = "Text"          # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None
    gender: str = "any"
    brand_contains: Optional[str] = None
    budget: float = 350.0
    num_outfits: int = 5
    articles: int = 3                  
    per_cat_candidates: int = 2

    def __post_init__(self):
        if self.budget is None:
            self.budget = 350.0

@dataclass
class Item:
    uuid: str
    score: Optional[float]
    rerank_score: Optional[float]
    properties: Dict[str, Any]

def _bytes_from_b64(img_b64: str | None) -> Optional[bytes]:
    if not img_b64:
        return None
    try:
        return base64.b64decode(img_b64.split(",")[-1])
    except Exception:
        return None

# ------------------------
# Clients & Models
# ------------------------
@lru_cache(maxsize=1)
def get_client() -> weaviate.WeaviateClient:
    if not WEAVIATE_URL or not WEAVIATE_API_KEY:
        raise RuntimeError("Missing WEAVIATE_HOST / WEAVIATE_API_KEY.")
    return weaviate.connect_to_weaviate_cloud(
        cluster_url=WEAVIATE_URL,
        auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    )

@lru_cache(maxsize=1)
def get_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    p = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return m, p, device

# Load FashionCLIP only if enabled
if ALLOW_FCLIP_RERANK:
    

    @lru_cache(maxsize=1)
    def get_fashionclip():
        device = "cuda" if torch.cuda.is_available() else "cpu"
        m = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
        p = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
        tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
        return m, p, device, tokenizer
else:
    get_fashionclip = None  

# ------------------------
# Embeddings
# ------------------------
@lru_cache(maxsize=4096)
def embed_text(text: str) -> np.ndarray:
    model, proc, device = get_clip()
    inputs = proc(text=[text or ""], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        vec = model.get_text_features(**inputs).cpu().numpy().flatten().astype(np.float32)
    n = np.linalg.norm(vec) + 1e-9
    return (vec / n).astype(np.float32)

def embed_texts_batch(texts: List[str]) -> np.ndarray:
    model, proc, device = get_clip()
    inputs = proc(text=texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        vecs = model.get_text_features(**inputs).cpu().numpy().astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
    return (vecs / norms).astype(np.float32)

@lru_cache(maxsize=1024)
def embed_image(image_bytes: bytes) -> np.ndarray:
    model, proc, device = get_clip()
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        vec = model.get_image_features(**inputs).cpu().numpy().flatten().astype(np.float32)
    n = np.linalg.norm(vec) + 1e-9
    return (vec / n).astype(np.float32)

# ------------------------
# Filters & helpers
# ------------------------
def build_filters(
    gender: Optional[str],
    categories: Optional[List[str]],
    #price_min: Optional[float],
    price_max: Optional[float],
    brand_substr: Optional[str],
    exclude_ids: Optional[List[str]] = None,
) -> Optional[Filter]:
    parts = []
    parts.append(Filter.by_property("gender").equal(gender))
    if categories:
        parts.append(Filter.any_of([Filter.by_property("category").equal(c) for c in categories]))
    # if price_min is not None:
    #     parts.append(Filter.by_property("price").greater_or_equal(price_min))
    if price_max is not None:
        parts.append(Filter.by_property("price").less_or_equal(price_max))
    # if brand_substr:
    #     parts.append(Filter.by_property("brand").like(f"*{brand_substr}*"))
    if exclude_ids:
        try:
            parts.append(Filter.by_id().not_in(exclude_ids))
        except Exception:
            pass
    return Filter.all_of(parts) if parts else None

def _compose_rerank_text(p: Dict) -> str:
    return " | ".join([
        str(p.get("title","")),
        str(p.get("description",""),
        str(p.get("image_caption","")),
        )[:400],
    ])

def lightweight_rerank(objs, query_text: str, query_img_vec: Optional[np.ndarray]):
    if not objs: return []
    texts = [_compose_rerank_text(o.properties or {}) for o in objs]
    cand_vecs = embed_texts_batch(texts)  # (n,d)
    q = (query_img_vec.reshape(1,-1) if query_img_vec is not None else embed_text(query_text).reshape(1,-1))
    sims = (q @ cand_vecs.T).flatten()
    boosts = []
    qt = (query_text or "").lower()
    for o in objs:
        p = o.properties or {}
        b = 0.0
        for key in ("title","description","image_caption"):
            val = str(p.get(key,"")).lower()
            if val and val in qt:
                b += 0.02
        boosts.append(b)
    scores = sims + np.array(boosts, dtype=np.float32)
    order = np.argsort(-scores)
    return [objs[i] for i in order]

if ALLOW_FCLIP_RERANK:
    def fashionclip_rerank(objs, query_text: str, query_img_bytes: Optional[bytes]):
        if not objs: return []
        model, processor, device, tokenizer = get_fashionclip()
        texts = [_compose_rerank_text(o.properties or {}) for o in objs]
        text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        with torch.no_grad():
            doc_embs = model.get_text_features(**text_inputs)
        doc_embs = doc_embs / (doc_embs.norm(dim=1, keepdim=True) + 1e-9)

        if query_img_bytes is not None:
            try:
                query_img = Image.open(io.BytesIO(query_img_bytes)).convert("RGB")
                img_inputs = processor(images=query_img, return_tensors="pt").to(device)
                with torch.no_grad():
                    q_emb = model.get_image_features(**img_inputs)
            except Exception:
                q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    q_emb = model.get_text_features(**q_inputs)
        else:
            q_inputs = processor(text=[query_text or ""], return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                q_emb = model.get_text_features(**q_inputs)

        q_emb = q_emb / (q_emb.norm(dim=1, keepdim=True) + 1e-9)
        sims = (q_emb @ doc_embs.T).squeeze(0).cpu().numpy()
        order = np.argsort(-sims)
        for o, s in zip(objs, sims):
            try:
                if o.metadata: setattr(o.metadata, "fclip_score", float(s))
            except Exception:
                pass
        return [objs[i] for i in order]
else:
    def fashionclip_rerank(objs, query_text: str, query_img_bytes: Optional[bytes]):
        # Fallback silently to lightweight rerank
        return lightweight_rerank(objs, query_text, None)

def _serialize_item(obj) -> Item:
    props = obj.properties or {}
    score = getattr(getattr(obj, "metadata", None), "score", None)
    rr = getattr(getattr(obj, "metadata", None), "fclip_score", None)
    return Item(uuid=obj.uuid, score=score, rerank_score=rr, properties=props)

def _normalize_price(x) -> float:
    try:
        if x is None: return 0.0
        if isinstance(x, (int,float)): return float(x)
        return float(str(x).replace("€","").replace("$","").replace(",","").strip())
    except Exception:
        return 0.0

def _score_rank_order(index: int, total: int) -> float:
    if total <= 1: return 1.0
    return 1.0 - (index / (total - 1))

def canonicalize_category(s: str) -> Optional[str]:
    if not s: return None
    k = str(s).strip()
    if k in CATEGORY_OPTIONS: return k
    return CAT_SYNONYMS.get(k.lower(), None)

# ------------------------
# Core retrieval
# ------------------------
def do_hybrid_search(
    text_query: str,
    text_vec: Optional[np.ndarray],
    image_vec: Optional[np.ndarray],
    limit: int,
    #offset: int,
    filters: Optional[Filter],
):
    client = get_client()
    if image_vec is not None:
        effective_alpha = 0.60
        primary_vec = image_vec
        target_vector = "image_vector"
    else:
        effective_alpha = 0.00
        primary_vec = text_vec
        target_vector = "text_vector"
    logger.debug("Running hybrid search with query=%s, limit=%s, filters=%s",
                  text_query, limit, filters)
    q = client.collections.get(COLLECTION).query.hybrid(
        query=text_query or "",
        vector=primary_vec.tolist() if primary_vec is not None else None,
        alpha=float(effective_alpha),
        target_vector=target_vector,
        query_properties=QUERY_PROPS,
        limit=int(limit),
        #offset=int(offset),
        filters=filters,
        return_properties=RETURN_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    logger.debug("Got %d objects back", len(q.objects) if q.objects else 0)
    return q.objects

# ------------------------
# Public API: Single items
# ------------------------
def retrieve_single_items(p: SingleSearchParams) -> Dict[str, Any]:

    cats = p.categories or []
    price_max = p.budget
    img_bytes = _bytes_from_b64(p.image_b64)
    text_vec = embed_text(p.text_query) if p.search_with in ("Text","Text + Image") and p.text_query.strip() else None
    image_vec = embed_image(img_bytes) if img_bytes and p.search_with in ("Image","Text + Image") else None

    filters = build_filters(
        gender=p.gender,
        categories=cats,
        price_max=price_max,
        brand_substr=p.brand_contains,
        exclude_ids=p.exclude_ids,
    )

    objs = do_hybrid_search(
        text_query=p.text_query,
        text_vec=text_vec,
        image_vec=image_vec,
        limit=max(p.limit, p.topk_for_rerank),
        filters=filters,
    )

    if ALLOW_FCLIP_RERANK:
        if img_bytes:
            reranked = fashionclip_rerank(objs[:p.topk_for_rerank], p.text_query, img_bytes)
        else:
            reranked = fashionclip_rerank(objs[:p.topk_for_rerank], p.text_query, None)
    else:
        reranked = lightweight_rerank(objs[:p.topk_for_rerank], p.text_query, image_vec)

    final = reranked[:p.limit]
    return {
        "count": len(final),
        "items": [asdict(_serialize_item(o)) for o in final],
    }

# ------------------------
# Public API: Outfit composer
# ------------------------
OPTIONAL_CATEGORIES = {
    "Outerwear",
    "Accessories",
    "Jewelry & Watches",
    "Bags",
    "Small Leather Goods",
}

def _build_category_filters(
    cat: str,
    gender: str,
    budget: Optional[float],
    brand: Optional[str],
    cap: Optional[float]
) -> Optional[Filter]:
    """
    Build a category-specific filter using weighted budget allocation
    instead of static tier lookups.
    """
    # allocate budget share
    weight = OUTFIT_WEIGHTS.get(cat, 0.2)  # fallback if not in weights
    hi = (budget * weight) if budget is not None else None
    lo = hi * 0.3 if hi is not None else None  # lo = 30% of hi for reasonable floor

    # apply user-specified cap
    if cap is not None:
        if hi is None:
            hi = cap
        else:
            hi = min(hi, cap)

    # fix inverted ranges
    if hi is not None and lo is not None and hi < lo:
        logger.debug(f" Adjusting inverted bounds for {cat}: ({lo}, {hi}) → ({lo}, {cap})")
        hi = cap if cap is not None else hi
        if hi < lo:  # still invalid → drop lo
            lo = None

    logger.debug(
        f"Category '{cat}' budget bounds: {lo} - {hi} (cap: {cap}) "
        f"| (budget: {budget}, weight: {weight})"
    )

    return build_filters(
        gender=None if gender == "any" else gender,
        categories=[cat],
        price_max=hi,
        brand_substr=(brand or None),
    )


def _compute_category_caps(cats_in_outfit, total_budget):
    weights = []
    for c in cats_in_outfit:
        w = OUTFIT_WEIGHTS.get(c, 1.0)
        weights.append(w)
    s = sum(weights) or 1.0
    return {c: float(total_budget * (w / s)) for c, w in zip(cats_in_outfit, weights)}

def call_llm_plan_safe(event, categories, budget, style_prefs = "", num_outfits=1):
        """Try OpenAI planner; fall back to None if not configured/usable."""
        try:
            # Prefer OpenAI v1 client if available
            try:
                from openai import OpenAI
                _client = OpenAI()
                resp = _client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You output only valid JSON: a LIST of outfit objects keyed by your provided categories."},
                        {"role": "user", "content": f"""
Create {num_outfits} outfit plans as a JSON LIST. Use these exact categories only:
{categories}

Constraints:
- Event: {event}
- Total budget (EUR): {budget}
- Style/brand hints: {style_prefs}

Output example (LIST):
[
  {{"Tops": "white oxford shirt, slim fit", "Bottoms": "navy tailored trousers", "Shoes": "black leather oxford shoes"}}
]
"""}
                    ],
                    temperature=0.5,
                )
                raw = resp.choices[0].message.content.strip()
            except Exception:
                # Older SDKs: try module-level call; still fully optional
                raw = None

            if not raw:
                return None

            # Parse + canonicalize
            data = json.loads(raw)
            if isinstance(data, dict) and "outfits" in data:
                data = data["outfits"]
            if isinstance(data, dict):
                data = [data]

            plans = []
            for outfit in (data or []):
                if not isinstance(outfit, dict):
                    continue
                plan = {}
                for k, v in outfit.items():
                    cat = canonicalize_category(k) or k
                    if cat in categories:
                        plan[cat] = v if isinstance(v, str) else " ".join(v) if isinstance(v, list) else str(v)
                # Ensure we only keep requested categories in order
                plan = {c: plan.get(c, f"{event} {c} {style_prefs or ''}".strip()) for c in categories}
                plans.append(plan)
            return plans or None
        except Exception:
            return None
        
def _lm_fallback_plan(num_outfits, cats, text_query, brand_hint):
    plans = []
    base = (text_query or "").strip()
    for _ in range(num_outfits):
        plan = {}
        for c in cats:
            if "top" in c.lower():  # only apply global theme to Tops
                q = " ".join([w for w in [base, c, (brand_hint or "").strip()] if w])
            else:
                q = " ".join([w for w in ["", c, (brand_hint or "").strip()] if w])
            plan[c] = q
        plans.append(plan)
    return plans

# ------------ OUTFIT BEAM SEARCH ------------

# def compose_outfits(p: OutfitParams, beam_width: int = 10) -> Dict[str, Any]:
#     """
#     Compose outfits using beam search instead of full itertools.product to handle
#     multiple categories and per-category candidates efficiently.
#     """
#     cats = OUTFIT_ORDER_BY_N[5 if p.articles >= 5 else p.articles]
#     plan = call_llm_plan_safe(
#         event=p.text_query,
#         categories=cats,
#         budget=p.budget,
#         num_outfits=p.num_outfits,
#     ) or _lm_fallback_plan(p.num_outfits, cats, p.text_query, p.brand_contains)

#     img_bytes = _bytes_from_b64(p.image_b64)
#     image_vec = embed_image(img_bytes) if img_bytes and p.search_with in ("Image", "Text + Image") else None

#     rerank_fn = (lambda objs, q: fashionclip_rerank(objs, q, img_bytes)) if ALLOW_FCLIP_RERANK \
#                 else (lambda objs, q: lightweight_rerank(objs, q, image_vec))

#     used_ids = set()
#     valid_outfits, near_misses, all_outfits = [], [], []

#     for outfit_idx in range(min(p.num_outfits, len(plan))):
#         outfit_plan = plan[outfit_idx]
#         cats_in_this = list(outfit_plan.keys())
#         total_budget = p.budget if p.budget is not None else 350.0
#         caps = _compute_category_caps(cats_in_this, total_budget)

#         candidates_map, missing_cats = {}, []
#         for cat in cats_in_this:
#             f = _build_category_filters(cat, p.gender, p.budget, p.brand_contains, caps.get(cat))
#             objs = do_hybrid_search(
#                 text_query=(outfit_plan.get(cat) or f"{p.text_query} {cat} {p.brand_contains}".strip()),
#                 text_vec=None, image_vec=None,
#                 limit=max(6, p.per_cat_candidates * 2),
#                 filters=f,
#             )
#             reranked = rerank_fn(objs[:max(6, p.per_cat_candidates * 2)], outfit_plan.get(cat, ""))
#             # precompute price
#             for o in reranked:
#                 try:
#                     o._price = _normalize_price((o.properties or {}).get("price", 0))
#                 except Exception:
#                     o._price = 0.0
#             filtered = [o for o in reranked if o.uuid not in used_ids][:p.per_cat_candidates]
#             if not filtered:
#                 missing_cats.append(cat)
#                 continue
#             candidates_map[cat] = filtered

#         if not candidates_map:
#             logger.debug(f"Skipping outfit {outfit_idx}, all categories empty")
#             continue

#         cand_lists = list(candidates_map.values())
#         cats_in_this = list(candidates_map.keys())

#         # precompute rank scores
#         rank_scores = []
#         for lst in cand_lists:
#             n = max(1, len(lst))
#             rank_scores.append({o.uuid: _score_rank_order(i, n) for i, o in enumerate(lst)})

#         # -------- BEAM SEARCH IMPLEMENTATION --------
#         beam = [([], 0.0, 0.0)]  # list of tuples: (partial_combo, score_so_far, price_so_far)

#         for cat_idx, lst in enumerate(cand_lists):
#             new_beam = []
#             for partial, score_so_far, price_so_far in beam:
#                 for o in lst:
#                     if o.uuid in {x.uuid for x in partial}:
#                         continue  # skip duplicates
#                     cap = caps.get(cats_in_this[cat_idx]) or (total_budget / len(cats_in_this))
#                     s_rank = rank_scores[cat_idx][o.uuid]
#                     s_price_penalty = 0.02 * (o._price / max(1.0, cap))
#                     new_score = score_so_far + (s_rank - s_price_penalty)
#                     new_price = price_so_far + o._price
#                     new_beam.append((partial + [o], new_score, new_price))
#             # keep only top beam_width sequences
#             beam = sorted(new_beam, key=lambda x: (x[1], -x[2] if x[2] <= total_budget else -1e12), reverse=True)[:beam_width]

#         # pick best valid combo under budget
#         best_combo, best_score, best_price = None, -1e9, 1e12
#         for combo, score, total_price in beam:
#             if total_price <= total_budget and score > best_score:
#                 best_combo, best_score, best_price = combo, score, total_price

#         # fallback: allow over-budget near-misses
#         if best_combo is None:
#             best_over, best_over_score, best_gap = None, -1e9, 1e12
#             for combo, score, total_price in beam:
#                 gap = total_price - total_budget
#                 if gap >= 0 and (gap < best_gap or (gap == best_gap and score > best_over_score)):
#                     best_gap, best_over_score, best_over = gap, score, combo
#             if best_over:
#                 combo = best_over
#                 total_price = sum(o._price for o in combo)
#                 for o in combo: used_ids.add(o.uuid)
#                 out_items = [asdict(_serialize_item(o)) for o in combo]
#                 near_misses.append({"items": out_items, "total_price": total_price, "missing_categories": missing_cats})
#                 all_outfits.append({"items": out_items, "total_price": total_price, "missing_categories": missing_cats})
#             continue

#         # commit valid combo
#         for o in best_combo: used_ids.add(o.uuid)
#         out_items = [asdict(_serialize_item(o)) for o in best_combo]
#         valid_outfits.append({"items": out_items, "total_price": best_price, "missing_categories": missing_cats})
#         all_outfits.append({"items": out_items, "total_price": best_price, "missing_categories": missing_cats})

#     return {"valid_outfits": valid_outfits, "near_misses": near_misses, "composed": all_outfits}

def compose_outfits(p: OutfitParams) -> Dict[str, Any]:
    """
    Compose outfits by directly combining fetched items from each category.
    No reranking, no beam search, no HuggingFace dependencies.
    """
    # Determine categories based on number of articles
    cats = OUTFIT_ORDER_BY_N[5 if p.articles >= 5 else p.articles]
    plan = call_llm_plan_safe(
        event=p.text_query,
        categories=cats,
        budget=p.budget,
        num_outfits=p.num_outfits,
    ) or _lm_fallback_plan(p.num_outfits, cats, p.text_query, p.brand_contains)

    used_ids = set()
    valid_outfits, near_misses, all_outfits = [], [], []

    for outfit_idx in range(min(p.num_outfits, len(plan))):
        outfit_plan = plan[outfit_idx]
        cats_in_this = list(outfit_plan.keys())
        total_budget = p.budget if p.budget is not None else 350.0
        caps = _compute_category_caps(cats_in_this, total_budget)

        candidates_map, missing_cats = {}, []

        # Fetch candidates per category
        for cat in cats_in_this:
            f = _build_category_filters(cat, p.gender, p.budget, p.brand_contains, caps.get(cat))
            objs = do_hybrid_search(
                text_query=(outfit_plan.get(cat) or f"{p.text_query} {cat} {p.brand_contains}".strip()),
                text_vec=None,
                image_vec=None,
                limit=p.num_outfits,  # fetch exactly the number of outfits
                filters=f,
            )
            # Filter out already used items
            filtered = [o for o in objs if o.uuid not in used_ids]
            if not filtered:
                missing_cats.append(cat)
                continue
            candidates_map[cat] = filtered[:p.num_outfits]  # take top N candidates

        if not candidates_map:
            logger.debug(f"Skipping outfit {outfit_idx}, all categories empty")
            continue

        # Generate outfits by simple positional combination
        for i in range(p.num_outfits):
            combo, total_price = [], 0.0
            for cat in candidates_map.keys():
                items = candidates_map[cat]
                # wrap around if fewer candidates than outfits
                o = items[i % len(items)]
                combo.append(o)
                total_price += _normalize_price((o.properties or {}).get("price", 0))
                used_ids.add(o.uuid)

            out_items = [asdict(_serialize_item(o)) for o in combo]
            valid_outfits.append({"items": out_items, "total_price": total_price, "missing_categories": missing_cats})
            all_outfits.append({"items": out_items, "total_price": total_price, "missing_categories": missing_cats})

    return {"valid_outfits": valid_outfits, "near_misses": near_misses, "composed": all_outfits}



# ------------------------
# Convenience dispatcher
# ------------------------
def retrieve(
    mode: str,
    single: Optional[SingleSearchParams] = None,
    outfit: Optional[OutfitParams] = None
) -> Dict[str, Any]:
    if mode.lower() in ("single","single items","single_items"):
        if not single: raise ValueError("single params required for mode=single")
        return retrieve_single_items(single)
    elif mode.lower() in ("outfit","outfit builder","outfit_builder"):
        if not outfit: raise ValueError("outfit params required for mode=outfit")
        return compose_outfits(outfit)
    else:
        raise ValueError(f"Unknown mode: {mode}")