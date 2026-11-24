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
    "brown sugar": [],
    "powdered sugar": [],
    "black pepper": ["paminta", "ground black pepper", "peppercorn", "black peppercorn",
                     "whole peppercorn", "ground pepper", "=pepper"],
    "white pepper": [],
    "cooking oil": ["mantika", "vegetable oil", "canola oil", "corn oil", "=oil"],
    "olive oil": [],
    "sesame oil": [],
    "coconut oil": [],
    "butter": ["mantikilya"],
    "margarine": [],
    "cheese": ["keso"],
    "kesong puti": ["white cheese"],
    "milk": ["gatas"],
    "evaporated milk": [],
    "condensed milk": [],
    "all purpose cream": [],
    "coconut milk": ["gata"],
    "coconut cream": ["kakang gata"],
    "coconut water": [],
    "water": ["tubig"],
    "flour": ["harina", "all purpose flour"],
    "rice flour": [],
    "cornstarch": ["corn starch"],
    "rice": ["bigas", "kanin", "steamed rice", "cooked rice", "white rice"],
    "rice noodles": [],
    "sotanghon": ["glass noodle", "cellophane noodle", "bean thread noodle", "vermicelli noodle"],
    "miki noodles": ["miki noodle"],
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
    "snow peas": [],
    "green peas": [],
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
                     "chili", "sili", "=chili"],
    "bird's eye chili": ["siling labuyo", "labuyo", "thai chili"],
    "long green pepper": ["siling haba", "siling pangsigang", "finger chili", "long chili",
                          "long green chili"],
    "bell pepper": ["red bell pepper", "green bell pepper", "yellow bell pepper", "capsicum"],
    "red pepper flakes": [],
    "cayenne pepper": [],
    "paprika": [],
    "hot sauce": [],
    "bean sprouts": ["togue", "taugeh", "mung bean sprouts"],
    "mung bean": ["monggo", "mongo"],
    "tofu": ["tokwa"],
    "mushroom": ["button mushroom", "shiitake mushroom", "whole mushroom"],
    "truffle": [],
    "okra": [],
    "spinach": ["baby spinach"],
    "water chestnut": [],
    "celery": [],
    "cucumber": [],
    "green papaya": [],
    "kamias": [],
    "young corn": [],
    "banana": [],
    "saba banana": ["saging na saba", "saba"],
    "banana ketchup": ["banana catsup", "banana sauce"],
    "banana blossom": ["puso ng saging"],
    "plantain": [],
    "jackfruit": ["langka", "nangka"],
    "pineapple": ["pinya"],
    "pineapple juice": [],
    "mango": ["mangga"],
    "raisin": ["pasas"],
    "coconut": [],
    "green olives": [],
    "pickle relish": [],
    "honey": [],
    "kinchay": [],

    # --- meat, poultry, eggs and seafood ------------------------------------
    "pork": ["baboy"],
    "pork_belly": ["pork belly", "liempo", "pork liempo"],
    "pork_hock": ["pork hock", "pork hocks", "pata"],
    "ground_pork": ["ground pork", "giniling", "pork giniling"],
    "pork_ribs": ["pork ribs"],
    "pork_shoulder": ["pork shoulder"],
    "pork_loin": ["pork loin"],
    "chicken": ["manok"],
    "chicken_breast": ["chicken breast", "chicken fillet"],
    "chicken_thigh": ["chicken thigh", "chicken thighs"],
    "whole chicken": [],
    "beef": ["baka"],
    "beef shank": ["bulalo"],
    "beef tripe": ["goto"],
    "egg": ["itlog"],
    "pork blood": ["dugo ng baboy"],
    "liver": ["atay"],
    "liver spread": [],
    "bacon": [],
    "ham": [],
    "longganisa": ["chinese sausage"],
    "chorizo de bilbao": [],
    "chicharon": ["chicharron", "pork cracklings"],
    "lechon": [],
    "milkfish": ["bangus", "bangus milkfish"],
    "tilapia": [],
    "salmon": [],
    "tuna": [],
    "lapu lapu": [],
    "shrimp": ["hipon", "prawn"],
    "squid": ["pusit"],
    "mussel": ["tahong"],
    "crab": ["alimango", "alimasag"],
    "anchovy": ["dilis"],
    "dried fish": ["tuyo"],
    "tinapa flakes": ["tinapa"],
    "fish flakes": [],

    # --- prepared ingredients: a dish used as an ingredient ------------------
    "leftover_pork_adobo": [
        "leftover pork adobo", "leftover adobo", "leftover adobo meat",
        "adobo flakes", "shredded adobo flakes from leftover adobo",
        "adobo flakes from leftover adobo", "=adobo flakes",
    ],
    "leftover_chicken": ["leftover chicken", "leftover chicken flakes"],
    "leftover_adobo_sauce": ["leftover adobo sauce", "seasoned vinegar"],
    "adobo": ["pork adobo", "chicken adobo"],

    # --- other packaged items ------------------------------------------------
    "mayonnaise": ["mayo"],
    "lumpia wrapper": ["lumpia wrappers"],
    "bread crumbs": [],
    "curry powder": [],
    "ground cumin": [],
    "turmeric": ["turmeric powder", "2 turmeric"],
    "worcestershire sauce": [],
    "hoisin sauce": [],
    "mirin": [],
    "wine": [],
    "star anise": [],
    "cilantro": ["coriander leaves", "wansoy"],
    "parsley": [],
    "peanut": ["mani"],
    "peanut butter": [],
    "sesame seed": [],
}

# Parent links let retrieval match a generic pantry item to a more specific
# recipe ingredient (e.g. pantry "pork" matches a recipe using "pork belly").
PARENTS: dict[str, str] = {
    "pork_belly": "pork",
    "pork_hock": "pork",
    "ground_pork": "pork",
    "pork_ribs": "pork",
    "pork_shoulder": "pork",
    "pork_loin": "pork",
    "chicken_breast": "chicken",
    "chicken_thigh": "chicken",
    "leftover_pork_adobo": "pork",
    "beef shank": "beef",
    "beef tripe": "beef",
    "chicharon": "pork",
    "lechon": "pork",
    "leftover_chicken": "chicken",
    "saba banana": "banana",
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
