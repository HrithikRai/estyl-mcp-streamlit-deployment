from __future__ import annotations
import io, json, time, itertools, base64
from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Optional, List, Dict, Tuple, Any

import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import Filter, MetadataQuery

from .config import (
    WEAVIATE_URL, WEAVIATE_API_KEY, COLLECTION,
    QUERY_PROPS, RETURN_PROPS, BUDGET_TIERS, CATEGORY_OPTIONS,
    OUTFIT_WEIGHTS, OUTFIT_ORDER_BY_N, CAT_SYNONYMS,
)

import logging
logging.basicConfig(level=logging.DEBUG)

# ------------------------
# Dataclasses / Types
# ------------------------
@dataclass
class SingleSearchParams:
    text_query: str = ""
    search_with: str = "Text"          # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None    # base64 encoded image
    gender: str = "any"                # "any" | "male" | "female" | "unisex"
    categories: List[str] = None
    brand_contains: Optional[str] = None
    budget_tier: str = "Mid"
    budget: float = 350.0
    limit: int = 12
    topk_for_rerank: int = 40
    offset: int = 0
    exclude_ids: Optional[List[str]] = None

@dataclass
class OutfitParams:
    text_query: str = ""               # event / vibe / style hints
    search_with: str = "Text"          # "Text" | "Image" | "Text + Image"
    image_b64: Optional[str] = None
    gender: str = "any"
    brand_contains: Optional[str] = None
    budget_tier: str = "Mid"
    budget: float = 350.0
    num_outfits: int = 3
    articles: int = 3                  # 2..5
    per_cat_candidates: int = 5

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

@lru_cache(maxsize=1)
def get_fashionclip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip").to(device)
    p = CLIPProcessor.from_pretrained("patrickjohncyh/fashion-clip")
    return m, p, device

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
    price_min: Optional[float],
    price_max: Optional[float],
    brand_substr: Optional[str],
    exclude_ids: Optional[List[str]] = None,
) -> Optional[Filter]:
    parts = []
    if gender and gender.lower() != "any":
        parts.append(Filter.by_property("gender").equal(gender))
    if categories:
        parts.append(Filter.any_of([Filter.by_property("category").equal(c) for c in categories]))
    if price_min is not None:
        parts.append(Filter.by_property("price").greater_or_equal(price_min))
    if price_max is not None:
        parts.append(Filter.by_property("price").less_or_equal(price_max))
    if brand_substr:
        parts.append(Filter.by_property("brand").like(f"*{brand_substr}*"))
    if exclude_ids:
        try:
            parts.append(Filter.by_id().not_in(exclude_ids))
        except Exception:
            pass
    return Filter.all_of(parts) if parts else None

def _compose_rerank_text(p: Dict) -> str:
    return " | ".join([
        str(p.get("title","")),
        str(p.get("brand","")),
        str(p.get("category","")),
        str(p.get("color","")),
        str(p.get("image_caption","")),
        str(p.get("unique_features","")),
        str(p.get("description",""))[:400],
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
        for key in ("category","color","brand"):
            val = str(p.get(key,"")).lower()
            if val and val in qt:
                b += 0.02
        boosts.append(b)
    scores = sims + np.array(boosts, dtype=np.float32)
    order = np.argsort(-scores)
    return [objs[i] for i in order]

def fashionclip_rerank(objs, query_text: str, query_img_bytes: Optional[bytes]):
    if not objs: return []
    model, processor, device = get_fashionclip()
    texts = [_compose_rerank_text(o.properties or {}) for o in objs]

    tokenizer = CLIPTokenizer.from_pretrained("patrickjohncyh/fashion-clip")
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
    # attach score if possible
    for o, s in zip(objs, sims):
        try:
            if o.metadata: setattr(o.metadata, "fclip_score", float(s))
        except Exception:
            pass
    return [objs[i] for i in order]

def _serialize_item(obj) -> Item:
    props = obj.properties or {}
    score = getattr(getattr(obj, "metadata", None), "score", None)
    rr = getattr(getattr(obj, "metadata", None), "fclip_score", None)
    return Item(uuid=obj.uuid, score=score, rerank_score=rr, properties=props)

def _price_bounds_for(cat: str, tier: str) -> Tuple[Optional[float], Optional[float]]:
    if cat not in BUDGET_TIERS: return None, None
    return BUDGET_TIERS[cat][tier]

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
    offset: int,
    filters: Optional[Filter],
):
    client = get_client()
    if image_vec is not None:
        effective_alpha = 0.60
        primary_vec = image_vec
        target_vector = "image_vector"
    else:
        effective_alpha = 0.0
        primary_vec = text_vec
        target_vector = "text_vector"
    logging.debug("Running hybrid search with query=%s, limit=%s, filters=%s",
                  text_query, limit, filters)
    q = client.collections.get(COLLECTION).query.hybrid(
        query=text_query or "",
        vector=primary_vec.tolist() if primary_vec is not None else None,
        alpha=float(effective_alpha),
        target_vector=target_vector,
        query_properties=QUERY_PROPS,
        limit=int(limit),
        offset=int(offset),
        filters=filters,
        return_properties=RETURN_PROPS,
        return_metadata=MetadataQuery(score=True),
    )
    logging.debug("Got %d objects back", len(q.objects) if q.objects else 0)
    return q.objects

# ------------------------
# Public API: Single items
# ------------------------
def retrieve_single_items(p: SingleSearchParams) -> Dict[str, Any]:
    cats = p.categories or []
    price_min, price_max = None, None

    # Single category → use tier bounds; else cap by overall budget
    if len(cats) == 1 and cats[0] in BUDGET_TIERS:
        lo, hi = BUDGET_TIERS[cats[0]][p.budget_tier]
        price_min, price_max = lo, hi
        if p.budget is not None:
            price_max = min(price_max, p.budget) if price_max else p.budget
    else:
        if p.budget is not None:
            price_max = p.budget

    img_bytes = _bytes_from_b64(p.image_b64)
    text_vec = embed_text(p.text_query) if p.search_with in ("Text","Text + Image") and p.text_query.strip() else None
    image_vec = embed_image(img_bytes) if img_bytes and p.search_with in ("Image","Text + Image") else None

    filters = build_filters(
        gender=None if p.gender == "any" else p.gender,
        categories=cats,
        price_min=price_min,
        price_max=price_max,
        brand_substr=p.brand_contains,
        exclude_ids=p.exclude_ids,
    )

    objs = do_hybrid_search(
        text_query=p.text_query,
        text_vec=text_vec,
        image_vec=image_vec,
        limit=max(p.limit, p.topk_for_rerank),
        offset=p.offset,
        filters=filters,
    )

    # Rerank (prefer FashionCLIP for fashion domain)
    if img_bytes:
        reranked = fashionclip_rerank(objs[:p.topk_for_rerank], p.text_query, img_bytes)
    else:
        reranked = fashionclip_rerank(objs[:p.topk_for_rerank], p.text_query, None)

    final = reranked[:p.limit]
    return {
        "count": len(final),
        "items": [asdict(_serialize_item(o)) for o in final],
    }

# ------------------------
# Public API: Outfit composer
# ------------------------
def _build_category_filters(cat: str, gender: str, tier: str, brand: Optional[str], cap: Optional[float]) -> Optional[Filter]:
    lo, hi = _price_bounds_for(cat, tier)
    if cap is not None:
        hi = min(hi, cap) if hi is not None else cap
    return build_filters(
        gender=None if gender == "any" else gender,
        categories=[cat],
        price_min=lo, price_max=hi,
        brand_substr=(brand or None),
    )

def _compute_category_caps(cats_in_outfit, total_budget):
    weights = []
    for c in cats_in_outfit:
        w = OUTFIT_WEIGHTS.get(c, 1.0)
        weights.append(w)
    s = sum(weights) or 1.0
    return {c: float(total_budget * (w / s)) for c, w in zip(cats_in_outfit, weights)}

def _lm_fallback_plan(num_outfits, cats, text_query, brand_hint):
    plans = []
    base = (text_query or "").strip()
    for _ in range(num_outfits):
        plan = {}
        for c in cats:
            q = " ".join([w for w in [base, c, (brand_hint or "").strip()] if w])
            plan[c] = q
        plans.append(plan)
    return plans

def compose_outfits(p: OutfitParams) -> Dict[str, Any]:
    # Determine categories by article count
    cats = OUTFIT_ORDER_BY_N[5 if p.articles >= 5 else p.articles]
    plan = _lm_fallback_plan(p.num_outfits, cats, p.text_query, p.brand_contains)

    img_bytes = _bytes_from_b64(p.image_b64)
    used_ids = set()
    valid_outfits, near_misses, all_outfits = [], [], []

    for outfit_idx in range(min(p.num_outfits, len(plan))):
        outfit_plan = plan[outfit_idx]
        cats_in_this = list(outfit_plan.keys())
        total_budget = p.budget if p.budget is not None else 350.0
        caps = _compute_category_caps(cats_in_this, total_budget)

        # retrieve per-category candidates
        candidates_map = {}
        for cat in cats_in_this:
            f = _build_category_filters(cat, p.gender, p.budget_tier, p.brand_contains, caps.get(cat))
            objs = do_hybrid_search(
                text_query=(outfit_plan.get(cat) or f"{p.text_query} {cat} {p.brand_contains}".strip()),
                text_vec=None, image_vec=None,
                limit=max(6, p.per_cat_candidates * 2),
                offset=0,
                filters=f,
            )
            if img_bytes:
                reranked = fashionclip_rerank(objs[:max(8, p.per_cat_candidates*2)], outfit_plan.get(cat, ""), img_bytes)
            else:
                reranked = lightweight_rerank(objs[:max(8, p.per_cat_candidates*2)], outfit_plan.get(cat, ""), None)

            filtered = [o for o in reranked if o.uuid not in used_ids][:p.per_cat_candidates]
            candidates_map[cat] = filtered

        if any(len(candidates_map.get(c, [])) == 0 for c in cats_in_this):
            continue

        cand_lists = [candidates_map[c] for c in cats_in_this]
        best_combo = None
        best_score = -1e9
        best_price = 1e12

        # precompute rank scores
        rank_scores = []
        for lst in cand_lists:
            n = max(1, len(lst))
            rank_scores.append({o.uuid: _score_rank_order(i, n) for i, o in enumerate(lst)})

        def _item_price(o):
            return _normalize_price((o.properties or {}).get("price", 0))

        for combo in itertools.product(*cand_lists):
            uuids = {o.uuid for o in combo}
            if len(uuids) < len(combo):  # guard dupes
                continue
            total_price = sum(_item_price(o) for o in combo)
            score = 0.0
            for idx, (cat, o) in enumerate(zip(cats_in_this, combo)):
                s_rank = rank_scores[idx][o.uuid]
                cap = caps.get(cat) or (p.budget / len(cats_in_this))
                s_price_penalty = 0.02 * (_item_price(o) / max(1.0, cap))
                score += (s_rank - s_price_penalty)

            if total_price <= p.budget and (score > best_score or (score == best_score and total_price < best_price)):
                best_score, best_combo, best_price = score, combo, total_price

        if best_combo is None:
            # choose closest over budget
            best_over, best_gap, best_over_score = None, 1e12, -1e9
            for combo in itertools.product(*cand_lists):
                uuids = {o.uuid for o in combo}
                if len(uuids) < len(combo): continue
                total_price = sum(_item_price(o) for o in combo)
                gap = total_price - p.budget
                score = 0.0
                for idx, (cat, o) in enumerate(zip(cats_in_this, combo)):
                    s_rank = rank_scores[idx][o.uuid]
                    cap = caps.get(cat) or (p.budget / len(cats_in_this))
                    s_price_penalty = 0.02 * (_item_price(o) / max(1.0, cap))
                    score += (s_rank - s_price_penalty)
                if gap < best_gap or (gap == best_gap and score > best_over_score):
                    best_gap, best_over_score, best_over = gap, score, (combo, total_price)
            if best_over:
                combo, total_price = best_over
                for o in combo: used_ids.add(o.uuid)
                out_items = [asdict(_serialize_item(o)) for o in combo]
                near_misses.append({"items": out_items, "total_price": total_price})
                all_outfits.append({"items": out_items, "total_price": total_price})
            continue

        final_list = list(best_combo)
        for o in final_list: used_ids.add(o.uuid)
        out_items = [asdict(_serialize_item(o)) for o in final_list]
        valid_outfits.append({"items": out_items, "total_price": best_price})
        all_outfits.append({"items": out_items, "total_price": best_price})

    return {
        "valid_outfits": valid_outfits,
        "near_misses": near_misses,
        "composed": all_outfits,
    }

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
