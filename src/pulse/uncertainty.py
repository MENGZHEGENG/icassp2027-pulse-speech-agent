from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import exp, log
import re
from statistics import mean

try:
    import pronouncing as _pronouncing
except ModuleNotFoundError:  # pragma: no cover - optional pronunciation backend
    _pronouncing = None


@dataclass(frozen=True)
class Hypothesis:
    text: str
    score: float | None = None


@dataclass(frozen=True)
class UncertaintySummary:
    top_text: str
    entropy: float | None
    instability: float
    unique_hypotheses: int
    entity_risk: float
    phonetic_risk: float = 0.0


ARPABET_LEXICON: dict[str, tuple[str, ...]] = {
    "account": ("AH", "K", "AW", "N", "T"),
    "an": ("AE", "N"),
    "c": ("S", "IY"),
    "eva": ("IY", "V", "AH"),
    "for": ("F", "AO", "R"),
    "four": ("F", "AO", "R"),
    "geneva": ("JH", "AH", "N", "IY", "V", "AH"),
    "jane": ("JH", "EY", "N"),
    "jen": ("JH", "EH", "N"),
    "john": ("JH", "AA", "N"),
    "liu": ("L", "Y", "UW"),
    "louis": ("L", "UW", "IH", "S"),
    "meera": ("M", "IH", "R", "AH"),
    "mira": ("M", "IH", "R", "AH"),
    "n": ("EH", "N"),
    "nrc": ("EH", "N", "AA", "R", "S", "IY"),
    "patel": ("P", "AH", "T", "EH", "L"),
    "q": ("K", "Y", "UW"),
    "queue": ("K", "Y", "UW"),
    "r": ("AA", "R"),
    "rc": ("AA", "R", "S", "IY"),
    "report": ("R", "IH", "P", "AO", "R", "T"),
    "route": ("R", "UW", "T"),
    "sean": ("SH", "AO", "N"),
    "shawn": ("SH", "AO", "N"),
    "smith": ("S", "M", "IH", "TH"),
    "way": ("W", "EY"),
    "wei": ("W", "EY"),
}


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9' ]+", " ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def edit_distance(reference_tokens: list[str], hypothesis_tokens: list[str]) -> int:
    previous = list(range(len(hypothesis_tokens) + 1))
    for reference_index, reference_token in enumerate(reference_tokens, start=1):
        current = [reference_index]
        for hypothesis_index, hypothesis_token in enumerate(hypothesis_tokens, start=1):
            deletion = previous[hypothesis_index] + 1
            insertion = current[hypothesis_index - 1] + 1
            substitution = previous[hypothesis_index - 1] + int(reference_token != hypothesis_token)
            current.append(min(deletion, insertion, substitution))
        previous = current
    return previous[-1]


def token_distance_ratio(reference_text: str, hypothesis_text: str) -> float:
    reference_tokens = normalize_text(reference_text).split()
    hypothesis_tokens = normalize_text(hypothesis_text).split()
    denominator = max(1, len(reference_tokens), len(hypothesis_tokens))
    return edit_distance(reference_tokens, hypothesis_tokens) / denominator


def phonetic_word_key(word: str) -> str:
    normalized = normalize_text(word).replace("'", "")
    if not normalized:
        return ""
    token = normalized.split()[0]
    replacements = [
        ("ough", "o"),
        ("augh", "af"),
        ("eigh", "a"),
        ("tion", "shun"),
        ("ph", "f"),
        ("sh", "s"),
        ("ch", "c"),
        ("ck", "k"),
        ("qu", "k"),
        ("x", "ks"),
        ("z", "s"),
        ("q", "k"),
        ("c", "k"),
        ("y", "i"),
    ]
    for source, target in replacements:
        token = token.replace(source, target)
    token = re.sub(r"(?<=[aeiou])w(?=[a-z]|$)", "", token)
    token = re.sub(r"(?<!^)h", "", token)
    token = re.sub(r"[aeiou]+", "A", token)
    token = re.sub(r"(.)\1+", r"\1", token)
    if token.startswith("A"):
        return "A" + token[1:].replace("A", "")
    return token[0] + token[1:].replace("A", "")


def phonetic_phrase_key(text: str) -> tuple[str, ...]:
    return tuple(key for token in normalize_text(text).split() if (key := phonetic_word_key(token)))


def strip_arpabet_stress(phone: str) -> str:
    return re.sub(r"\d", "", phone)


@lru_cache(maxsize=8192)
def cmudict_word_key(word: str) -> tuple[str, ...]:
    if _pronouncing is None:
        return ()
    token = normalize_text(word).replace("'", "").split()
    if not token:
        return ()
    phones = _pronouncing.phones_for_word(token[0])
    if not phones:
        return ()
    return tuple(strip_arpabet_stress(phone) for phone in phones[0].split())


def arpabet_word_key(word: str) -> tuple[str, ...]:
    token = normalize_text(word).replace("'", "").split()
    if not token:
        return ()
    return ARPABET_LEXICON.get(token[0]) or cmudict_word_key(token[0])


def arpabet_phrase_key(text: str) -> tuple[str, ...]:
    phones: list[str] = []
    for token in normalize_text(text).split():
        phones.extend(arpabet_word_key(token))
    return tuple(phones)


def joined_phonetic_phrase_key(text: str) -> str:
    return "".join(phonetic_phrase_key(text))


def phonetic_key_distance_ratio(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    left_chars = list("".join(left))
    right_chars = list("".join(right))
    denominator = max(1, len(left_chars), len(right_chars))
    return edit_distance(left_chars, right_chars) / denominator


def arpabet_key_distance_ratio(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    denominator = max(1, len(left), len(right))
    return edit_distance(list(left), list(right)) / denominator


def contains_phonetic_phrase(text: str, phrase_key: tuple[str, ...]) -> bool:
    if not phrase_key:
        return False
    text_keys = phonetic_phrase_key(text)
    width = len(phrase_key)
    if "".join(phrase_key) and "".join(phrase_key) in "".join(text_keys):
        return True
    if len(text_keys) < width:
        return False
    return any(text_keys[start : start + width] == phrase_key for start in range(len(text_keys) - width + 1))


def contains_arpabet_phrase(text: str, phrase_key: tuple[str, ...]) -> bool:
    if not phrase_key:
        return False
    text_keys = arpabet_phrase_key(text)
    width = len(phrase_key)
    if phrase_key and phrase_key == text_keys:
        return True
    if len(text_keys) < width:
        return False
    return any(text_keys[start : start + width] == phrase_key for start in range(len(text_keys) - width + 1))


def has_close_phonetic_candidates(term_keys: list[tuple[str, ...]], threshold: float = 0.5) -> bool:
    for left_index, left_key in enumerate(term_keys):
        for right_key in term_keys[left_index + 1 :]:
            if left_key and right_key and phonetic_key_distance_ratio(left_key, right_key) <= threshold:
                return True
    return False


def has_close_arpabet_candidates(term_keys: list[tuple[str, ...]], threshold: float = 0.35) -> bool:
    for left_index, left_key in enumerate(term_keys):
        for right_key in term_keys[left_index + 1 :]:
            if left_key and right_key and arpabet_key_distance_ratio(left_key, right_key) <= threshold:
                return True
    return False


def score_entropy(hypotheses: list[Hypothesis]) -> float | None:
    scores = [hypothesis.score for hypothesis in hypotheses if hypothesis.score is not None]
    if len(scores) != len(hypotheses) or not scores:
        return None
    max_score = max(scores)
    weights = [exp(score - max_score) for score in scores]
    total_weight = sum(weights)
    probability_by_text: dict[str, float] = {}
    for hypothesis, weight in zip(hypotheses, weights):
        key = normalize_text(hypothesis.text)
        probability_by_text[key] = probability_by_text.get(key, 0.0) + weight / total_weight
    probabilities = list(probability_by_text.values())
    entropy = -sum(probability * log(probability + 1e-12) for probability in probabilities)
    return max(0.0, entropy)


def consensus_instability(hypotheses: list[Hypothesis]) -> float:
    if len(hypotheses) <= 1:
        return 0.0
    top_text = hypotheses[0].text
    distances = [token_distance_ratio(top_text, hypothesis.text) for hypothesis in hypotheses[1:]]
    return mean(distances) if distances else 0.0


def entity_disagreement(hypotheses: list[Hypothesis], entity_terms: list[str] | None = None) -> float:
    if len(hypotheses) <= 1:
        return 0.0
    if not entity_terms:
        return 0.0
    normalized_terms = [normalize_text(term) for term in entity_terms]
    signatures: list[tuple[str, ...]] = []
    for hypothesis in hypotheses:
        normalized = normalize_text(hypothesis.text)
        present_terms = tuple(term for term in normalized_terms if term and term in normalized)
        signatures.append(present_terms)
    if len(signatures) >= 2 and signatures[0] != signatures[1]:
        return 1.0
    return (len(set(signatures)) - 1) / max(1, len(signatures) - 1)


def phonetic_entity_disagreement(hypotheses: list[Hypothesis], entity_terms: list[str] | None = None) -> float:
    if len(hypotheses) <= 1 or not entity_terms:
        return 0.0

    term_keys = [phonetic_phrase_key(term) for term in entity_terms]
    phone_term_keys = [arpabet_phrase_key(term) for term in entity_terms]
    signatures: list[frozenset[int]] = []
    for hypothesis in hypotheses:
        present_terms = frozenset(
            term_index
            for term_index, term_key in enumerate(term_keys)
            if contains_phonetic_phrase(hypothesis.text, term_key)
            or contains_arpabet_phrase(hypothesis.text, phone_term_keys[term_index])
        )
        signatures.append(present_terms)

    nonempty_signatures = [signature for signature in signatures if signature]
    if not signatures[0]:
        return 1.0
    if not nonempty_signatures:
        return 0.0
    if len(signatures) >= 2 and signatures[0] != signatures[1]:
        return 1.0

    shared_key_sizes: dict[tuple[str, ...], int] = {}
    for term_key in term_keys:
        if term_key:
            shared_key_sizes[term_key] = shared_key_sizes.get(term_key, 0) + 1
    ambiguous_term_present = any(
        shared_key_sizes.get(term_keys[term_index], 0) > 1
        for signature in nonempty_signatures
        for term_index in signature
    )
    if ambiguous_term_present or has_close_phonetic_candidates(term_keys) or has_close_arpabet_candidates(phone_term_keys):
        return 0.8

    return (len(set(nonempty_signatures)) - 1) / max(1, len(nonempty_signatures) - 1)


def summarize_hypotheses(
    hypotheses: list[Hypothesis],
    entity_terms: list[str] | None = None,
) -> UncertaintySummary:
    if not hypotheses:
        raise ValueError("At least one hypothesis is required")
    normalized_unique = {normalize_text(hypothesis.text) for hypothesis in hypotheses}
    phonetic_risk = phonetic_entity_disagreement(hypotheses, entity_terms)
    return UncertaintySummary(
        top_text=hypotheses[0].text,
        entropy=score_entropy(hypotheses),
        instability=consensus_instability(hypotheses),
        unique_hypotheses=len(normalized_unique),
        entity_risk=entity_disagreement(hypotheses, entity_terms),
        phonetic_risk=phonetic_risk,
    )
