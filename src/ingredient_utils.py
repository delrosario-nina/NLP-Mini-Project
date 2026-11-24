"""Shared helpers for the Pinoy recipe RAG project.

HOW TO USE
    import sys; sys.path.insert(0, "src")
    from alias_map import alias_map, parents_map
    from ingredient_utils import build_lookup, normalize_text, expand_canonicals
    lookup, collisions = build_lookup(alias_map())
    canonicals, leftover = normalize_text("leftover pork adobo", lookup)
    pantry_tokens = expand_canonicals(canonicals, parents_map())

Used by the alias-map notebook now, and later to normalize user pantry queries,
so recipes and queries go through exactly the same normalization. Compound
terms (pork belly, ground pork, Maggi Magic Sarap, leftover pork adobo) are
matched before their generic parts via longest-first matching.
"""
import re
import json
import unicodedata

UNITS = set("""
cup cups tablespoon tablespoons tbsp tbsps teaspoon teaspoons tsp tsps lb lbs pound pounds
ounce ounces oz g gram grams kg kilo kilos kilogram kilograms ml l liter liters litre litres
piece pieces clove cloves bunch bunches bundle bundles can cans pinch pinches thumb thumbs
head heads stalk stalks sprig sprigs slice slices pack packs packet packets sachet sachets
stick sticks inch inches quart quarts gallon gallons dash handful handfuls
""".split())

PREP = set("""
chopped minced sliced diced cut cleaned washed peeled crushed grated cubed cubes julienned
wedged wedges halved quartered boiled cooked fried shredded mashed trimmed rinsed drained
thinly finely roughly coarsely optional divided softened melted beaten soaked removed
deveined seeded pounded flaked cored strips strip chunks chunk serving servings garnish
garnishing pcs pc stemmed ends
""".split())

STOPWORDS = set("""
of and or to taste for the a an with into from about fresh some more extra big small medium
large thick thin whole inch lengthwise pieces optional plus if needed as desired
one two three four five six seven eight nine ten half
""".split())

DROP = UNITS | PREP | STOPWORDS


def line_to_phrases(line):
    """Turn a raw ingredient line into one or more cleaned ingredient phrases.

    '4 cloves Garlic (peeled), chopped'      -> ['garlic']
    'Fish sauce and ground black pepper to taste' -> ['fish sauce', 'ground black pepper']
    """
    s = unicodedata.normalize("NFKD", str(line).lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))   # chicharrón -> chicharron, jalapeño -> jalapeno
    s = re.sub(r"\([^)]*\)", " ", s)
    parts = re.split(r"\band\b|&", s) if "to taste" in s else [s]
    phrases = []
    for part in parts:
        for chunk in part.split(","):
            chunk = re.sub(r"[^a-z\s'-]", " ", chunk).replace("-", " ")
            toks = [t for t in chunk.split() if t not in DROP and len(t) > 1]
            if toks:
                phrases.append(" ".join(toks))
    return phrases


def build_lookup(alias_map):
    """alias_map: {canonical: [aliases]} -> ({term: canonical}, [collision messages])."""
    lookup, collisions = {}, []
    for canon, aliases in alias_map.items():
        canon_n = canon.strip().lower()
        for term in [canon_n] + [a.strip().lower() for a in aliases]:
            if term.startswith("="):          # exact-phrase-only alias, e.g. "=pepper"
                term = "=" + term[1:].strip()
            if term in lookup:
                if lookup[term] != canon_n:
                    collisions.append(f"'{term}' -> '{lookup[term]}' and '{canon_n}'")
                continue  # keep the first definition; collisions must not flip silently
            lookup[term] = canon_n
    return lookup, collisions


def expand_canonicals(canonicals, parents):
    """Add parent ingredients so a generic pantry item can satisfy a specific cut.

    Example: canonical ``pork_belly`` also yields ``pork`` when ``parents`` maps
    ``pork_belly -> pork``.
    """
    out = []
    for canon in canonicals:
        for value in (canon, parents.get(canon)):
            if value and value not in out:
                out.append(value)
    return out


def _variants(token):
    out = [token]
    if token.endswith("ves") and len(token) > 4:
        out.append(token[:-3] + "f")          # leaves -> leaf
    if token.endswith("ies") and len(token) > 4:
        out.append(token[:-3] + "y")          # anchovies -> anchovy
    if token.endswith("es") and len(token) > 3:
        out.append(token[:-2])
    if token.endswith("s") and len(token) > 2:
        out.append(token[:-1])
    return out


def match_canonicals(phrase, lookup, max_n=6):
    """Greedy longest-match of known terms inside a phrase.

    ``max_n=6`` lets multi-word compounds such as "adobo flakes from leftover
    adobo" match as one term instead of degrading to generic "pork".

    Aliases written with a leading "=" (e.g. "=pepper") only match when the whole
    phrase equals them, so "pepper" -> black pepper but "hot pepper" does not.

    Returns (canonicals_found_in_order, leftover_tokens_not_covered).
    """
    if "=" + phrase in lookup:               # exact-phrase-only alias
        return [lookup["=" + phrase]], []
    toks = phrase.split()
    covered = [False] * len(toks)
    found = {}
    for n in range(min(max_n, len(toks)), 0, -1):
        for start in range(len(toks) - n + 1):
            if any(covered[start:start + n]):
                continue
            head = toks[start:start + n - 1]
            for last in _variants(toks[start + n - 1]):
                key = " ".join(head + [last])
                if key in lookup:
                    found[start] = lookup[key]
                    for k in range(start, start + n):
                        covered[k] = True
                    break
    canonicals = []
    for start in sorted(found):
        if found[start] not in canonicals:
            canonicals.append(found[start])
    leftover = [t for t, c in zip(toks, covered) if not c]
    return canonicals, leftover


def normalize_text(text, lookup):
    """Normalize a raw ingredient line or user query item -> (canonicals, leftover_tokens)."""
    canon, left = [], []
    for ph in line_to_phrases(text):
        c, l = match_canonicals(ph, lookup)
        for x in c:
            if x not in canon:
                canon.append(x)
        left += l
    return canon, left


def load_alias_map(path="data/processed/alias_map.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
