"""Query understanding for the Pinoy Recipe RAG project.

HOW TO USE
    import sys; sys.path.insert(0, "src")
    from alias_map import parents_map
    from ingredient_utils import build_lookup, load_alias_map
    from query_understanding import merge_constraints, accumulate_pantry

    lookup, _ = build_lookup(load_alias_map())
    texts = ["sitaw, bawang, i only have half an hour", "actually im a vegetarian now"]
    constraints = merge_constraints(lookup, texts)     # diet/time/type/allergens
    pantry, unmapped = accumulate_pantry(lookup, texts) # canonicals across messages

This module is the MECHANISM layer of the project — it lives in ``src/`` next
to the alias map and the dish groups, so the notebooks (04-style pipeline
analysis) and the web interface share exactly the same tokenisation and
constraint resolution:

    parse_constraints   one message -> diet, time budget, dish type, allergies
    merge_constraints   whole conversation -> resolved state, most recent
                        statement wins ("no longer a vegetarian" clears the tag,
                        a new time budget replaces the old one, allergies
                        accumulate because one does not talk oneself out of them)
    accumulate_pantry   canonical pantry across all user messages
    expand_pantry       canonicals plus their generic parents (kasim -> pork)
    pantry_changed      did the latest message update the pantry/constraints?
    apply_constraints   filter retrieved recipes by diet/time/type/allergens
    rank_by_pantry_fit  order candidates by pantry fit (most matched, fewest
                        missing, fastest, then retrieval score)

The interface in ``webapp/`` imports this module; it does not define its own
copy. Any improvement here immediately benefits both entry points.
"""

from __future__ import annotations

import re
from typing import Sequence

from dish_groups import ingredient_importance, version_facts
from ingredient_utils import expand_canonicals, parse_query

# --- time budgets -----------------------------------------------------------

_TIME_NUMBERS = {
    "half": 0.5, "an": 1, "a": 1, "isang": 1,
    "one": 1, "una": 1, "two": 2, "dalawang": 2, "three": 3, "tatlong": 3,
    "four": 4, "apat": 4, "five": 5, "limang": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "fifteen": 15, "twenty": 20,
    "thirty": 30, "forty": 40, "forty-five": 45, "forty five": 45,
    "sixty": 60, "ninety": 90,
}
_TIME_UNITS_HOUR = {"hour", "hours", "hr", "hrs", "oras"}
_NUMBER = r"\d+(?:\.\d+)?|" + "|".join(_TIME_NUMBERS)
_TIME_RE = re.compile(
    rf"\b({_NUMBER})\s*(hour|hours|hr|hrs|oras|minute|minutes|min|mins|minuto|minutos)\b",
    flags=re.IGNORECASE,
)
_HALF_HOUR_RE = re.compile(
    r"\bhalf\s+(?:an?\s+)?(?:hour|oras)\b|kalahating\s+oras", flags=re.IGNORECASE
)
_TIME_CLEAR_RE = re.compile(
    r"no\s+time\s+limit|any\s+amount\s+of\s+time|time\s+(?:doesn'?t|does\s+not)\s+matter"
    r"|no\s+rush|as\s+long\s+as\s+(?:it\s+takes|needed|you\s+want)"
    r"|kahit\s+gaano\s+katagal",
    re.IGNORECASE,
)


def _minutes_in(text: str) -> int | None:
    """The time budget stated in ONE message, or None if it states none."""
    lowered = str(text).lower()
    minutes: list[float] = []
    for value, unit in _TIME_RE.findall(lowered):
        number = _TIME_NUMBERS.get(value.lower())
        if number is None:
            try:
                number = float(value)
            except ValueError:
                continue
        multiplier = 60 if unit.lower() in _TIME_UNITS_HOUR else 1
        minutes.append(number * multiplier)
    if _HALF_HOUR_RE.search(lowered):
        minutes.append(30.0)
    return int(min(minutes)) if minutes else None


# --- diets and lifestyles ---------------------------------------------------

_DIET_RES: list[tuple[str, re.Pattern]] = [
    ("vegan", re.compile(r"\bvegan\b|plant.?based|walang\s+karne", re.IGNORECASE)),
    (
        "pescatarian",
        re.compile(
            r"\bpescatarian\b|\bpescetarian\b|pesco.?vegetarian|\bpesco\b", re.IGNORECASE
        ),
    ),
    (
        "vegetarian",
        re.compile(
            r"\bvegetarian\b|\bveggie\b|\bvegy\b|meatless|no\s+meat|walang\s+karneng"
            r"|gulay\s+lang|vegetables\s+only",
            re.IGNORECASE,
        ),
    ),
]

# Negations come first: "no longer a vegetarian" still contains the word
# "vegetarian", so the positive patterns must not run when a negation matches.
_DIET_NEGATION = re.compile(
    r"no\s+longer\s+(?:a\s+)?(?:vegetarian|vegan|pescatarian|pescetarian|veggie|veg)\b"
    r"|not\s+(?:a\s+)?(?:vegetarian|vegan|pescatarian|pescetarian|veggie|veg)\b"
    r"|never\s+(?:been\s+)?(?:a\s+)?(?:vegetarian|vegan)\b"
    r"|quit\s+being\s+(?:a\s+)?(?:vegetarian|vegan|pescetarian)"
    r"|stopped\s+being\s+(?:a\s+)?(?:vegetarian|vegan)"
    r"|hindi\s+(?:na\s+)?(?:ako\s+)?(?:vegetarian|vegan|veggie|veg)\b"
    r"|\beat(?:ing)?\s+meat\s+(?:again|now)\b"
    r"|\bi\s+(?:can|do)\s+eat\s+meat\b"
    r"|kumakain\s+na\s+(?:ulit\s+)?ng\s+karne",
    re.IGNORECASE,
)

# --- dish types -------------------------------------------------------------

# Food art: bentos, decorated rice balls, centrepieces. They are real rows in
# the corpus but they are not dishes someone asks for by type — "Chicken
# Tinola Onigiri Bento" is not a soup even though "tinola" is in the name.
NOVELTY_NAME_RE = re.compile(
    r"bento|onigiri|charaben|kyaraben|decoration|centerpiece|food\s+art"
    r"|cake\s+pops?|shaped\s+like|cute\s+(?:chick|food)",
    re.I,
)


def is_novelty(name) -> bool:
    """True for decorated/food-art recipes (bento boxes, centrepieces)."""
    return bool(NOVELTY_NAME_RE.search(str(name)))


DISH_CATEGORIES: dict[str, tuple[re.Pattern, re.Pattern]] = {
    "soup": (
        re.compile(r"\bsoup\b|\bsabaw\b|\bsinigang\b|\bnilaga\b|\bbroth\b", re.I),
        re.compile(
            r"soup|sinigang|nilaga|sabaw|bulalo|batchoy|tinola|binacol|arroz caldo"
            r"|lugaw|goto\b|congee|mami\b|sopas",
            re.I,
        ),
    ),
    "stew": (
        re.compile(
            r"\bstew\b|\bestofado\b|\bkaldereta\b|\bcaldereta\b|\bkare.?kare\b|\bginataan\b",
            re.I,
        ),
        re.compile(
            r"stew|kaldereta|caldereta|kare|afritada|mechado|humba|estofado|ginataan"
            r"|sarciado|pochero|puchero|morcon|binagoongan|bopis|dinuguan",
            re.I,
        ),
    ),
    "breakfast": (
        re.compile(r"\bbreakfast\b|\balmusal\b|\bsilog\b|\bmorning\b", re.I),
        re.compile(
            r"silog|breakfast|champorado|lugaw|goto\b|arroz caldo|congee|tocino"
            r"|\btapa\b|daing|tuyo|longanisa|longganisa|saluyot",
            re.I,
        ),
    ),
    "noodles": (
        re.compile(
            r"\bnoodles?\b|\bpancit\b|\bpasta\b|\bspaghetti\b|\bmami\b|\blomi\b"
            r"|\bsotanghon\b|\bbihon\b|\bcanton\b",
            re.I,
        ),
        re.compile(
            r"pancit|noodle|spaghetti|pasta|mami\b|lomi|sotanghon|bihon|canton"
            r"|carbonara|palabok|malabon|misua|pad thai|chow mein",
            re.I,
        ),
    ),
    "fried rice": (
        re.compile(r"fried\s+rice|\bsinangag\b", re.I),
        re.compile(r"fried\s+rice|sinangag|java\s+rice|garlic\s+rice", re.I),
    ),
    "grilled": (
        re.compile(r"\bgrill(?:ed|s)?\b|\binasal\b|\bbbq\b|\bbarbecue\b|\binihaw\b", re.I),
        re.compile(r"grill|inasal|bbq|barbecue|inihaw", re.I),
    ),
    "dessert": (
        re.compile(
            r"\bdessert\b|\bpanghimagas\b|\bmerienda\b|\bsnack\b|\bsweets?\b|\bkakanin\b",
            re.I,
        ),
        re.compile(
            r"dessert|halaya|flan|cake|bibingka|\bbiko\b|kutsinta|pichi|leche|ginataan"
            r"|minatamis|carioca|turon|banana\s+cue|camote\s+cue|maruya|palitaw|sapin"
            r"|buko\s+salad|fruit\s+salad|halo.?halo|ice\s+cream|pudding|polvoron"
            r"|pastillas|yema|ginataang|suman|puto\b|kutsinta|ensaymada|spanish\s+bread",
            re.I,
        ),
    ),
    "salad": (
        re.compile(r"\bsalad\b|\bensalada\b", re.I),
        re.compile(r"salad|ensalada|appetizer|atsara", re.I),
    ),
}

# --- canonical exclusion sets per diet --------------------------------------
# Diet order: vegan > vegetarian > pescatarian.

_MEAT = {
    "pork", "pork_belly", "pork_hock", "ground_pork", "pork_ribs", "pork_shoulder",
    "pork_loin", "pork intestine", "pig ears", "kidney", "pork blood", "liver",
    "liver spread", "bacon", "ham", "longganisa", "chorizo de bilbao", "chicharon",
    "lechon", "leftover_pork_adobo", "adobo", "beef", "beef shank", "beef tripe",
    "goat meat", "corned beef", "hotdog", "pork bouillon", "beef bouillon",
}
_POULTRY = {
    "chicken", "chicken_breast", "chicken_thigh", "whole chicken",
    "leftover_chicken", "chicken bouillon",
}
_SEAFOOD = {
    "milkfish", "fish", "tilapia", "salmon", "tuna", "lapu lapu", "galunggong",
    "tulingan", "tanigue", "pompano", "sardines", "dried fish", "tinapa flakes",
    "fish flakes", "shrimp", "squid", "mussel", "crab", "clams", "anchovy",
    "fish sauce", "shrimp paste", "fish powder",
}
_ANIMAL = {
    "egg", "quail egg", "salted egg", "milk", "evaporated milk", "condensed milk",
    "cheese", "kesong puti", "butter", "margarine", "all purpose cream",
    "mayonnaise", "honey",
}
DIET_EXCLUSIONS = {
    "vegan": _MEAT | _POULTRY | _SEAFOOD | _ANIMAL,
    "vegetarian": _MEAT | _POULTRY | _SEAFOOD,
    "pescatarian": _MEAT | _POULTRY,
}

# --- allergies --------------------------------------------------------------

# Allergy phrases -> banned canonicals. Groups expand words that are not
# themselves a canonical ingredient ("seafood allergy" bans the seafood set).
_ALLERGEN_RE = re.compile(
    r"(?:allergic|allergy)(?:\s+ako)?\s+(?:to\s+|sa\s+)?"
    r"([a-z][a-z\s'-]{1,60}?)(?=\bbut\b|[,.;!?]|$)",
    re.IGNORECASE,
)
_ALLERGEN_SUFFIX_RE = re.compile(
    r"([a-z][a-z\s]{1,30}?)\s+allerg(?:y|ic)", re.IGNORECASE
)
ALLERGEN_GROUPS: dict[str, set] = {
    "seafood": _SEAFOOD,
    "shellfish": {"shrimp", "squid", "crab", "mussel", "clams"},
    "dairy": {
        "milk", "evaporated milk", "condensed milk", "cheese", "kesong puti",
        "butter", "margarine", "all purpose cream",
    },
    "egg": {"egg", "quail egg", "salted egg"},
    "peanut": {"peanut", "peanut butter"},
}


def _allergen_bans(phrase: str, lookup: dict) -> list[str]:
    """Turn an allergy phrase into canonical names to ban."""
    phrase = phrase.strip().lower()
    bans: list[str] = []
    canon, _ = parse_query(phrase, lookup)
    bans.extend(canon)
    for word, group in ALLERGEN_GROUPS.items():
        if word in phrase:
            for item in group:
                if item not in bans:
                    bans.append(item)
    return bans


# --- per-message and whole-conversation parsing ------------------------------

def parse_constraints(text: str, lookup: dict | None = None) -> dict:
    """Read diet, time budget, dish type, and allergies from ONE message.

    Returns per-message findings; merge_constraints resolves them across the
    conversation. Negations are checked first: "no longer a vegetarian" still
    contains the word "vegetarian", so the positive patterns must not run.
    """
    lowered = str(text).lower()

    diet = None
    diet_negated = False
    if _DIET_NEGATION.search(lowered):
        diet_negated = True
    else:
        for name, pattern in _DIET_RES:
            if pattern.search(lowered):
                diet = name
                break

    category = None
    for name, (query_re, _name_re) in DISH_CATEGORIES.items():
        if query_re.search(lowered):
            category = name
            break

    allergens: list[str] = []
    if lookup is not None:
        for match in _ALLERGEN_RE.finditer(lowered):
            for ban in _allergen_bans(match.group(1), lookup):
                if ban not in allergens:
                    allergens.append(ban)
        for match in _ALLERGEN_SUFFIX_RE.finditer(lowered):
            for ban in _allergen_bans(match.group(1), lookup):
                if ban not in allergens:
                    allergens.append(ban)

    return {
        "diet": diet,
        "diet_negated": diet_negated,
        "max_minutes": _minutes_in(lowered),
        "time_cleared": bool(_TIME_CLEAR_RE.search(lowered)),
        "category": category,
        "allergens": allergens,
    }


def merge_constraints(lookup: dict, texts: Sequence[str]) -> dict:
    """Resolve constraints across the conversation, most recent statement wins.

    Diets and time budgets are REPLACED by the latest statement (and can be
    cleared by "no longer a vegetarian" / "no time limit"), so old tags never
    linger. Allergies accumulate: they are not something one talks oneself
    out of between messages.
    """
    diet: str | None = None
    max_minutes: int | None = None
    category: str | None = None
    allergens: list[str] = []

    for text in texts:
        found = parse_constraints(str(text or ""), lookup)
        if found["diet_negated"]:
            diet = None
        elif found["diet"]:
            diet = found["diet"]
        if found["time_cleared"]:
            max_minutes = None
        elif found["max_minutes"] is not None:
            max_minutes = found["max_minutes"]
        if found["category"]:
            category = found["category"]
        for ban in found["allergens"]:
            if ban not in allergens:
                allergens.append(ban)

    return {
        "diet": diet,
        "max_minutes": max_minutes,
        "category": category,
        "allergens": allergens,
    }


# --- pantry -----------------------------------------------------------------

def accumulate_pantry(lookup: dict, texts: Sequence[str]) -> tuple[list[str], list[str]]:
    """Union of canonical ingredients mentioned across ALL user messages.

    A conversation's pantry is not frozen at the first message: when the user
    later says "actually i now have pork as well", pork joins the pantry so
    ranking, matched/missing facts, and the displayed card all reflect it.
    Unmapped tokens are only collected from messages that contributed at
    least one canonical, so plain questions add no noise.
    """
    canon: list[str] = []
    unmapped: list[str] = []
    for text in texts:
        found, left = parse_query(str(text), lookup)
        if not found:
            continue
        for item in found:
            if item not in canon:
                canon.append(item)
        for item in left:
            if item not in unmapped:
                unmapped.append(item)
    return canon, unmapped


def expand_pantry(parents: dict, canon: Sequence[str]) -> list[str]:
    """Pantry canonicals plus their generic parents (kasim -> pork shoulder, pork)."""
    return expand_canonicals(list(canon), parents)


def pantry_changed(
    lookup: dict, prior_texts: Sequence[str], all_texts: Sequence[str]
) -> bool:
    """True when the latest message added ingredients or changed constraints.

    Used to decide whether a follow-up is a plain question about the current
    recommendation or a pantry update that deserves a fresh recommendation.
    """
    prior_canon, _ = accumulate_pantry(lookup, prior_texts)
    all_canon, _ = accumulate_pantry(lookup, all_texts)
    if [c for c in all_canon if c not in prior_canon]:
        return True

    prior_constraints = merge_constraints(lookup, prior_texts)
    all_constraints = merge_constraints(lookup, all_texts)
    if len(all_constraints["allergens"]) > len(prior_constraints["allergens"]):
        return True
    return (
        prior_constraints["diet"] != all_constraints["diet"]
        or prior_constraints["max_minutes"] != all_constraints["max_minutes"]
        or prior_constraints["category"] != all_constraints["category"]
    )


# --- filtering and ranking --------------------------------------------------

def apply_constraints(df, hits, pantry: Sequence[str], constraints: dict):
    """Filter retrieved recipes by diet, time, dish type, and allergies.

    ``df`` needs the ``canon_match`` column (canonicals + their parents), which
    the retrieval layer prepares. Returns (filtered_hits, report). If a filter
    would remove everything, it is skipped and flagged in the report so the
    caller can say so honestly.
    """
    report = {
        "diet": constraints["diet"],
        "max_minutes": constraints["max_minutes"],
        "category": constraints["category"],
        "allergens": list(constraints["allergens"]),
        "diet_fallback": False,
        "time_fallback": False,
        "category_fallback": False,
        "allergen_fallback": False,
    }
    kept = list(hits)

    # Order matters: the dish type is the user's stated intent, so it runs
    # BEFORE the time budget. "Soup under 30 minutes" with no such soup then
    # falls back to the fastest soup with an honest note, instead of quietly
    # replacing the soup with a quick non-soup dish.
    if constraints["diet"]:
        banned = DIET_EXCLUSIONS[constraints["diet"]]
        survivors = [
            hit for hit in kept if not (banned & df.iloc[hit[0]]["canon_match"])
        ]
        if survivors:
            kept = survivors
        else:
            report["diet_fallback"] = True

    if constraints["category"]:
        _, name_re = DISH_CATEGORIES[constraints["category"]]
        survivors = [
            hit
            for hit in kept
            if name_re.search(str(df.iloc[hit[0]]["recipe_name"]))
            and not is_novelty(df.iloc[hit[0]]["recipe_name"])
        ]
        if survivors:
            kept = survivors
        else:
            report["category_fallback"] = True

    if constraints["max_minutes"] is not None:
        limit = constraints["max_minutes"]

        def within(hit) -> bool:
            return version_facts(df.iloc[hit[0]], pantry)["budget_minutes"] <= limit

        survivors = [hit for hit in kept if within(hit)]
        if survivors:
            kept = survivors
        else:
            report["time_fallback"] = True

    if constraints["allergens"]:
        banned = set(constraints["allergens"])
        survivors = [
            hit for hit in kept if not (banned & df.iloc[hit[0]]["canon_match"])
        ]
        if survivors:
            kept = survivors
        else:
            report["allergen_fallback"] = True

    return kept, report


def category_positions(df, category: str) -> list[int]:
    """Row positions whose name marks them as this dish type.

    The dish type is the user's stated intent, so it defines the candidate
    universe for retrieval instead of acting as a post-filter on a pool that
    may not contain any soup at all ("soup" appears in recipe NAMES, which
    BM25 never sees, so a great dense match can be drowned by fusion).
    Novelty/food-art names (bento boxes) are excluded.
    """
    spec = DISH_CATEGORIES.get(category)
    if not spec:
        return []
    _, name_re = spec
    return [
        i for i, name in enumerate(df["recipe_name"])
        if name_re.search(str(name)) and not is_novelty(name)
    ]


def rank_by_pantry_fit(df, hits, pantry: Sequence[str]):
    """Order retrieved recipes by how well they use the user's pantry.

    Retrieval (hybrid BM25 + dense) decides which dishes are candidates; this
    decides which one to SHOW first: most (importance-weighted) matched
    ingredients, then fewest missing, then fastest, then retrieval score.
    So when two versions match the same pantry items, the one missing fewer
    ingredients wins.
    """
    def key(hit):
        pos, score = hit
        facts = version_facts(df.iloc[pos], pantry)
        matched_weight = sum(ingredient_importance(item) for item in facts["matched"])
        return (
            -matched_weight,
            is_novelty(df.iloc[pos]["recipe_name"]),
            facts["missing_cost"],
            facts["budget_minutes"],
            -score,
        )

    return sorted(hits, key=key)
