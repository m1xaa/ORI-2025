import re
from typing import Dict, List, Any, Set

from fontTools import unicodedata
from nltk import SnowballStemmer

FILLERS_SR: Set[str] = {
    "ovaj", "ono", "znači", "mislim", "pa", "eee", "uh", "eto", "hm", "aha",
    "ma", "a", "dakle", "onako", "ustvari", "ovome", "ovoga", "ovde", "ovdeka",
    "čekaj", "gle", "znaš", "vidi", "kao", "upravo", "realno", "bukvalno",
    "otprilike", "uglavnom", "takođe", "ono kao", "valjda", "možda", "brate",
    "mhm", "da", "ne", "ovde ono", "znači ono", "ovaj ono", "mislim ono",
    "pa ono", "čuj", "pazi", "brate mili", "brate brate", "brateee", "brateeee",
    "baš", "ono stvarno", "znači bukvalno", "tako da", "mislim da"
}

FILLERS_EN: Set[str] = {
    "um", "uh", "er", "ah", "hmm", "huh", "yeah", "yep", "nope", "uhhuh", "mmm",
    "like", "you know", "so", "well", "okay", "ok", "alright", "right", "actually",
    "basically", "literally", "seriously", "honestly", "totally", "really",
    "kind of", "sort of", "kinda", "sorta", "i mean", "just", "maybe", "probably",
    "perhaps", "anyway", "anyways", "sure", "fine", "look", "listen",
    "you see", "i guess", "you know what i mean", "you know what i’m saying",
    "you feel me", "i dunno", "whatever", "stuff", "things", "like you know",
    "right so", "well so", "ok so", "yeah so", "oh well", "oh ok", "oh right",
    "by the way", "to be honest", "frankly", "basically like"
}

STEMMERS = {
    "en": SnowballStemmer("english"),
    "sr": SnowballStemmer("serbian")
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def normalize_case(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def remove_redundant_punctuation(text: str) -> str:
    return re.sub(r"\s([?.!,;:])", r"\1", text)


def normalize_text(text: str) -> str:
    text = normalize_whitespace(text)
    text = normalize_case(text)
    text = remove_redundant_punctuation(text)
    return text


def remove_fillers(text: str, lang: str) -> str:
    fillers = FILLERS_SR if lang.lower().startswith("sr") else FILLERS_EN

    for filler in sorted(fillers, key=len, reverse=True):
        pattern = rf"\b{re.escape(filler)}\b"
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    return normalize_whitespace(text)


def stem_text(text: str, lang: str) -> str:
    stemmer = STEMMERS.get(lang[:2])
    if not stemmer:
        return text
    tokens = re.findall(r"\w+", text)
    stemmed = [stemmer.stem(t) for t in tokens]
    return " ".join(stemmed)


def normalize_segments(segments: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for seg in segments:
        text = normalize_text(seg["text"])
        text = remove_fillers(text, lang=lang)
        text = stem_text(text, lang=lang)

        if text:
            results.append({
                "start": seg["start"],
                "end": seg["end"],
                "text": text
            })
    return results
