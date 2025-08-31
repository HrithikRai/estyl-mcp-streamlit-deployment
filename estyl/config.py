import os
from dotenv import load_dotenv

load_dotenv()

# Weaviate
WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")
COLLECTION = os.getenv("WEAVIATE_COLLECTION", "Estyl_articles")

# OpenAI (optional, for planner & host)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Query/Return props (kept identical to Streamlit app)
QUERY_PROPS = [
    "title","category","subcategory","subsubcategory","image_caption",
    "description","color","fabric","brand","unique_features"
]
RETURN_PROPS = [
    "title","gender","product_url","gcs_image_path","brand","price","category","color",
    "subcategory","subsubcategory","description","image_caption","unique_features"
]

CATEGORY_OPTIONS = [
    "Tops","Bottoms","Shoes","Outerwear","Jewelry & Watches","Accessories",
    "Activewear","Suits & Tailoring","Underwear","Bags","Dresses & One-Pieces","Lingerie/Underwear",
    "Small Leather Goods","Swimwear"
]

# Budget tiers (verbatim from your app)
BUDGET_TIERS = {
    "Tops": {"Budget": (0, 30), "Mid": (30, 70), "Premium": (70, 150), "Luxury": (150, None)},
    "Dresses & One-Pieces": {"Budget": (0, 100), "Mid": (100, 250), "Premium": (250, 600), "Luxury": (600, None)},
    "Bottoms": {"Budget": (0, 60), "Mid": (60, 130), "Premium": (130, 250), "Luxury": (250, None)},
    "Outerwear": {"Budget": (0, 120), "Mid": (120, 300), "Premium": (300, 700), "Luxury": (700, None)},
    "Suits & Tailoring": {"Budget": (0, 150), "Mid": (150, 350), "Premium": (350, 800), "Luxury": (800, None)},
    "Lingerie/Underwear": {"Budget": (0, 20), "Mid": (20, 50), "Premium": (50, 100), "Luxury": (100, None)},
    "Sleep & Lounge": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Activewear": {"Budget": (0, 40), "Mid": (40, 90), "Premium": (90, 180), "Luxury": (180, None)},
    "Swimwear": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Shoes": {"Budget": (0, 80), "Mid": (80, 150), "Premium": (150, 300), "Luxury": (300, None)},
    "Bags": {"Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)},
    "Small Leather Goods": {"Budget": (0, 40), "Mid": (40, 100), "Premium": (100, 200), "Luxury": (200, None)},
    "Accessories": {"Budget": (0, 30), "Mid": (30, 80), "Premium": (80, 150), "Luxury": (150, None)},
    "Jewelry & Watches": {"Budget": (0, 80), "Mid": (80, 200), "Premium": (200, 500), "Luxury": (500, None)},
}

OUTFIT_WEIGHTS = {"Tops": 0.22, "Bottoms": 0.25, "Shoes": 0.30, "Outerwear": 0.35, "Accessories": 0.08}

OUTFIT_ORDER_BY_N = {
    2: ["Tops", "Bottoms"],
    3: ["Tops", "Bottoms", "Shoes"],
    4: ["Tops", "Bottoms", "Shoes", "Outerwear"],
    5: ["Tops", "Bottoms", "Shoes", "Outerwear", "Accessories"],
}

CAT_SYNONYMS = {
    "tops": "Tops", "shirt": "Tops", "t-shirt": "Tops", "tee": "Tops", "blouse": "Tops",
    "bottoms": "Bottoms", "pants": "Bottoms", "trousers": "Bottoms", "jeans": "Bottoms", "skirt": "Bottoms", "shorts": "Bottoms",
    "shoes": "Shoes", "sneakers": "Shoes", "boots": "Shoes", "heels": "Shoes", "sandals": "Shoes", "loafers": "Shoes",
    "outerwear": "Outerwear", "jacket": "Outerwear", "coat": "Outerwear", "blazer": "Outerwear", "hoodie": "Outerwear",
    "accessories": "Accessories", "watch": "Accessories", "belt": "Accessories", "hat": "Accessories",
    "scarf": "Accessories", "sunglasses": "Accessories", "glasses": "Accessories", "tie": "Accessories",
}
