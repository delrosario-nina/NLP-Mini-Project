"""Maintained Filipino ingredient alias definitions.

HOW TO USE
    import sys; sys.path.insert(0, "src")
    from alias_map import alias_map, parents_map, write_alias_map
    from ingredient_utils import build_lookup, normalize_text

    aliases = alias_map()
    lookup, collisions = build_lookup(aliases)
    normalize_text("leftover pork adobo", lookup)   # -> ['leftover_pork_adobo']

    write_alias_map("data/processed/alias_map.json")   # for notebook 02/03/04

Why this file exists
    Notebook 02 defined ALIASES inline and patched it with several later
    ``ALIASES.update(...)`` calls, which made collisions and missing compound
    terms easy to introduce. This is the single source of truth. The notebook
    can still run; this module is the maintained version.
"""

from __future__ import annotations

import json
from pathlib import Path

# An alias starting with "=" only matches the whole phrase, e.g. "=pepper".
ALIASES: dict[str, list[str]] = {
    # --- pantry staples and sauces -----------------------------------------
    "soy sauce": ["toyo", "soya sauce", "dark soy sauce", "light soy sauce"],
    "fish sauce": ["patis"],
    "vinegar": ["suka", "cane vinegar", "white vinegar", "coconut vinegar", "vinegar white", "rice vinegar"],
    "salt": ["asin"],
    "sugar": ["asukal", "white sugar", "granulated sugar"],
    "brown sugar": ["muscovado", "dark brown sugar", "panutsa"],
    "powdered sugar": [],
    "black pepper": ["paminta", "ground black pepper", "peppercorn", "black peppercorn",
                     "whole peppercorn", "ground pepper", "=pepper"],
    "white pepper": ["pamintang puti", "white peppercorn", "ground white pepper"],
    "cooking oil": ["mantika", "vegetable oil", "canola oil", "corn oil", "=oil"],
    "olive oil": [],
    "sesame oil": [],
    "coconut oil": ["virgin coconut oil", "vco"],
    "butter": ["mantikilya"],
    "margarine": [],
    "cheese": ["keso", "cheddar", "cheddar cheese", "parmesan", "parmesan cheese",
               "quick melt", "quick melt cheese", "queso"],
    "kesong puti": ["white cheese"],
    "milk": ["gatas"],
    "evaporated milk": [],
    "condensed milk": [],
    "all purpose cream": ["all-purpose cream", "cooking cream"],
    "coconut milk": ["gata"],
    "coconut cream": ["kakang gata"],
    "coconut water": ["buko juice", "buko water"],
    "water": ["tubig"],
    "flour": ["harina", "all purpose flour"],
    "rice flour": ["galapong"],
    "cornstarch": ["corn starch"],
    "rice": ["bigas", "kanin", "steamed rice", "cooked rice", "white rice"],
    "rice noodles": ["bihon", "bihon noodles", "rice stick noodles", "rice vermicelli"],
    "sotanghon": ["glass noodle", "cellophane noodle", "bean thread noodle", "vermicelli noodle",
                  "sotanghon noodles"],
    "miki noodles": ["miki noodle", "canton", "pancit canton", "egg noodles"],
    "soda": ["sprite", "7-up", "lemon soda softdrink", "carbonated drink"],

    # --- Filipino seasoning mixes and bouillons -----------------------------
    "maggi magic sarap": ["magic sarap", "maggi", "=maggi magic sarap"],
    "liquid seasoning": ["knorr", "knorr liquid seasoning", "=liquid seasoning"],
    "sinigang mix": ["sinigang sa sampaloc mix", "knorr sinigang mix", "tamponang sinigang"],
    "ginataang gulay mix": ["knorr ginataang gulay mix"],
    "beef bouillon": ["beef cube", "beef powder", "beef broth", "beef stock"],
    "chicken bouillon": ["chicken cube", "chicken powder", "chicken broth", "chicken stock"],
    "pork bouillon": ["pork cube", "pork powder", "pork cube bouillon", "pork broth", "pork stock"],
    "annatto": ["atsuete", "achuete", "achiote", "annatto seed", "achuete seed",
                "annatto powder", "achuete oil", "annatto oil"],
    "tamarind": ["sampalok", "sampaloc"],
    "shrimp paste": ["bagoong alamang", "bagoong", "alamang"],
    "oyster sauce": [],
    "fish powder": [],
    "salted egg": ["itlog na maalat"],

    # --- aromatics, vegetables, fruit ---------------------------------------
    "garlic": ["bawang", "bawang na puti", "garlic powder"],
    "onion": ["sibuyas", "red onion", "white onion", "yellow onion"],
    "spring onion": ["green onion", "scallion", "dahon ng sibuyas"],
    "shallot": [],
    "leek": [],
    "ginger": ["luya"],
    "tomato": ["kamatis", "roma tomato"],
    "tomato sauce": [],
    "tomato paste": [],
    "string beans": ["sitaw", "snake bean", "long green bean", "yardlong bean",
                     "yard long bean", "long bean"],
    "green beans": ["baguio beans", "french beans"],
    "snow peas": ["chicharo", "sitsaro", "mangetout"],
    "green peas": ["gisantes", "sweet peas"],
    "water spinach": ["kangkong"],
    "eggplant": ["talong", "chinese eggplant"],
    "bitter melon": ["ampalaya", "bitter gourd"],
    "squash": ["kalabasa", "pumpkin"],
    "cabbage": ["repolyo", "napa cabbage"],
    "radish": ["labanos", "daikon"],
    "taro": ["gabi"],
    "sweet potato": ["camote", "kamote"],
    "chayote": ["sayote"],
    "moringa": ["malunggay", "moringa leaves", "malunggay leaves"],
    "bottle gourd": ["upo"],
    "bok choy": ["pechay", "petsay", "pak choi"],
    "mustard leaves": [],
    "potato": ["patatas"],
    "carrot": ["karot"],
    "calamansi": ["kalamansi", "calamondin", "calamansi juice"],
    "lemon": ["lemon juice"],
    "lime": ["lime juice"],
    "lemongrass": ["tanglad"],
    "bay leaf": ["dahon ng laurel", "laurel leaf", "laurel leaves", "dried bay leaf"],
    "chili pepper": ["red chili pepper", "green chili pepper", "red chili", "green chili",
                     "chili", "sili", "=chili", "siling pula", "pulang sili",
                     "siling berde", "berdeng sili"],
    "bird's eye chili": ["siling labuyo", "labuyo", "thai chili"],
    "long green pepper": ["siling haba", "siling pangsigang", "siling pansigang", "finger chili",
                          "long chili", "long green chili"],
    "bell pepper": ["red bell pepper", "green bell pepper", "yellow bell pepper", "capsicum",
                    "pimiento", "pimento"],
    "red pepper flakes": ["crushed red pepper"],
    "cayenne pepper": ["cayenne"],
    "paprika": ["smoked paprika"],
    "hot sauce": ["sriracha", "tabasco"],
    "bean sprouts": ["togue", "taugeh", "mung bean sprouts"],
    "mung bean": ["monggo", "mongo"],
    "tofu": ["tokwa"],
    "mushroom": ["button mushroom", "shiitake mushroom", "whole mushroom"],
    "truffle": [],
    "okra": [],
    "spinach": ["baby spinach"],
    "basil": ["basil leaves"],
    "cauliflower": ["cauliflower florets"],
    "lettuce": ["lettuce leaves"],
    "winged beans": ["winged bean", "sigarilyas"],
    "sponge gourd": ["patola"],
    "jicama": ["singkamas"],
    "water chestnut": [],
    "celery": [],
    "cucumber": ["pipino"],
    "green papaya": ["unripe papaya", "raw papaya", "papaya"],
    "kamias": ["bilimbi"],
    "young corn": [],
    "corn": ["sweet corn", "corn kernels"],
    "banana": ["saging", "ripe banana"],
    "saba banana": ["saging na saba", "saba"],
    "banana ketchup": ["banana catsup", "banana sauce"],
    "banana blossom": ["puso ng saging"],
    "plantain": [],
    "jackfruit": ["langka", "nangka"],
    "pineapple": ["pinya"],
    "pineapple juice": [],
    "mango": ["mangga"],
    "raisin": ["pasas"],
    "coconut": ["niyog", "grated coconut", "fresh coconut", "coconut meat"],
    "green olives": ["olives", "green olive"],
    "pickle relish": [],
    "honey": [],
    "kinchay": [],
    "pandan leaf": ["pandan leaves", "pandan"],
    "chickpea": ["chickpeas", "garbanzo beans", "garbanzos", "garbanzo"],

    # --- meat, poultry, eggs and seafood ------------------------------------
    "pork": ["baboy", "karne ng baboy"],
    "pork_belly": ["pork belly", "liempo", "pork liempo"],
    "pork_hock": ["pork hock", "pork hocks", "pata", "pigue", "pork pigue", "pata ng baboy"],
    "ground_pork": ["ground pork", "giniling", "pork giniling", "pork mince"],
    "pork_ribs": ["pork ribs", "ribs", "spare ribs", "spareribs", "pork spare ribs"],
    "pork_shoulder": ["pork shoulder", "kasim", "pork kasim", "picnic shoulder"],
    "pork_loin": ["pork loin", "pork chop", "pork chops", "pork lion"],
    "chicken": ["manok"],
    "chicken_breast": ["chicken breast", "chicken fillet"],
    "chicken_thigh": ["chicken thigh", "chicken thighs"],
    "whole chicken": [],
    "beef": ["baka", "beef sirloin", "sirloin", "beef brisket", "brisket", "beef chuck",
             "chuck", "beef tenderloin", "tenderloin", "beef ribs", "karne ng baka"],
    "beef shank": ["bulalo"],
    "beef tripe": ["goto"],
    "egg": ["itlog"],
    "pork blood": ["dugo ng baboy"],
    "liver": ["atay"],
    "liver spread": [],
    "bacon": [],
    "ham": [],
    "longganisa": ["longanisa", "chinese sausage"],
    "chorizo de bilbao": ["chorizo", "spanish chorizo"],
    "chicharon": ["chicharron", "pork cracklings"],
    "lechon": [],
    "milkfish": ["bangus", "bangus milkfish", "bangus na isda"],
    "fish": ["isda", "fresh fish", "whole fish", "isda fish"],
    "tilapia": ["tilapia fish"],
    "salmon": ["salmon fillet", "salmon steak"],
    "tuna": ["canned tuna", "tuna flakes"],
    "lapu lapu": ["lapu-lapu", "grouper", "lapu lapu fish"],
    "shrimp": ["hipon", "prawn", "hipon tagalog"],
    "squid": ["pusit", "calamari"],
    "mussel": ["tahong", "mussels"],
    "crab": ["alimango", "alimasag", "crabs"],
    "clams": ["clam", "halaan"],
    "anchovy": ["dilis", "dilis fish"],
    "dried fish": ["tuyo", "daing", "daing na isda", "bulad"],
    "tinapa flakes": ["tinapa", "tinapa fish", "smoked fish"],
    "fish flakes": ["fish flake", "flaked fish"],

    # --- more fish varieties seen in the corpus -------------------------------
    "galunggong": ["round scad"],
    "tulingan": ["frigate tuna", "bullet tuna"],
    "tanigue": ["tangigue", "spanish mackerel", "kingfish"],
    "pompano": [],
    "sardines": ["canned sardines", "sardine", "sardinas"],

    # --- prepared ingredients: a dish used as an ingredient ------------------
    "leftover_pork_adobo": [
        "leftover pork adobo", "leftover adobo", "leftover adobo meat",
        "adobo flakes", "shredded adobo flakes from leftover adobo",
        "adobo flakes from leftover adobo", "=adobo flakes",
        "leftover adobo flakes",
    ],
    "leftover_chicken": ["leftover chicken", "leftover chicken flakes"],
    "leftover_adobo_sauce": ["leftover adobo sauce", "seasoned vinegar"],
    "adobo": ["pork adobo", "chicken adobo"],

    # --- more ingredient vocabulary found in the corpus -----------------------
    "miso": ["miso paste", "white miso", "red miso"],
    "tausi": ["salted black beans", "fermented black beans"],
    "hotdog": ["hotdogs", "hot dog", "hot dogs", "filipino hotdog"],
    "quail egg": ["quail eggs", "itlog ng pugo"],
    "goat meat": ["kambing", "goat", "chevon"],
    "corned beef": ["canned corned beef"],
    "kidney": ["pork kidney", "beef kidney", "kidneys"],
    "pig ears": ["pig's ears", "pigs ears", "pig ear", "pig's ear", "pig ears"],
    "pork intestine": ["pig intestine", "pork intestines", "intestines", "isaw"],
    "misua": ["miswa", "mee sua", "wheat vermicelli"],
    "spaghetti": ["spaghetti noodles", "spaghetti pasta"],
    "bread": ["pandesal", "sliced bread", "bread loaf", "loaf bread"],
    "kaffir lime leaf": ["kaffir lime leaves", "kaffir"],

    # --- other packaged items ------------------------------------------------
    "mayonnaise": ["mayo"],
    "lumpia wrapper": ["lumpia wrappers"],
    "bread crumbs": ["breadcrumbs"],
    "baking powder": [],
    "spring roll wrappers": ["spring roll wrapper"],
    "lomi noodles": ["lomi noodle"],
    "curry powder": [],
    "cumin": ["ground cumin", "cumin powder"],
    "oregano": ["dried oregano", "oregano leaves"],
    "cinnamon": ["cinnamon stick", "cinnamon powder"],
    "ketchup": ["catsup", "tomato ketchup"],
    "turmeric": ["turmeric powder", "2 turmeric"],
    "worcestershire sauce": [],
    "hoisin sauce": [],
    "mirin": [],
    "wine": ["cooking wine", "rice wine", "sherry", "red wine", "white wine"],
    "star anise": [],
    "cilantro": ["coriander leaves", "wansoy", "coriander"],
    "parsley": [],
    "peanut": ["mani"],
    "peanut butter": [],
    "sesame seed": [],
}

# Parent links let retrieval match a generic pantry item to a more specific
# recipe ingredient (e.g. pantry "pork" matches a recipe using "pork belly"),
# and a specific pantry item fall back to its generic family when no exact
# dish exists (e.g. pantry "pork belly" still surfaces pork adobo).
PARENTS: dict[str, str] = {
    "pork_belly": "pork",
    "pork_hock": "pork",
    "ground_pork": "pork",
    "pork_ribs": "pork",
    "pork_shoulder": "pork",
    "pork_loin": "pork",
    "pork intestine": "pork",
    "pig ears": "pork",
    "kidney": "pork",
    "bacon": "pork",
    "ham": "pork",
    "longganisa": "pork",
    "chorizo de bilbao": "pork",
    "chicken_breast": "chicken",
    "chicken_thigh": "chicken",
    "leftover_pork_adobo": "pork",
    "leftover_chicken": "chicken",
    "beef shank": "beef",
    "beef tripe": "beef",
    "corned beef": "beef",
    "chicharon": "pork",
    "lechon": "pork",
    "saba banana": "banana",
    "quail egg": "egg",
    "milkfish": "fish",
    "tilapia": "fish",
    "salmon": "fish",
    "tuna": "fish",
    "lapu lapu": "fish",
    "galunggong": "fish",
    "tulingan": "fish",
    "tanigue": "fish",
    "pompano": "fish",
    "sardines": "fish",
    "dried fish": "fish",
    "tinapa flakes": "fish",
    "fish flakes": "fish",
}


def alias_map() -> dict[str, list[str]]:
    """Return a fresh copy so callers cannot mutate the module-level constant."""
    return {canonical: list(aliases) for canonical, aliases in ALIASES.items()}


def parents_map() -> dict[str, str]:
    return dict(PARENTS)


def write_alias_map(path: str | Path = "data/processed/alias_map.json") -> Path:
    """Write the alias map in the JSON shape notebook 02 already expects."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps({k: sorted(set(v)) for k, v in sorted(alias_map().items())},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return target
