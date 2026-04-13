import string
import unicodedata

_DIACRITIC_CHARS = frozenset("ʲː")
_PUNCTUATION = set(string.punctuation)


def is_diacritic(char: str) -> bool:
    return unicodedata.combining(char) != 0 or char in _DIACRITIC_CHARS


def split_into_phonemes(phonemes: str) -> list[str]:
    result: list[str] = []
    i = 0
    n = len(phonemes)

    while i < n:
        ch = phonemes[i]

        if ch.isspace():
            i += 1
            continue

        if ch == "(":
            group = "("
            i += 1
            while i < n and phonemes[i] != ")":
                if not phonemes[i].isspace():
                    group += phonemes[i]
                i += 1
            if i < n and phonemes[i] == ")":
                group += ")"
                i += 1
            result.append(group)
            continue

        if ch == "⁽":
            diacritic = ""
            i += 1
            while i < n and phonemes[i] != "⁾":
                if not phonemes[i].isspace():
                    diacritic += phonemes[i]
                i += 1
            if i < n and phonemes[i] == "⁾":
                i += 1
            suffix = "⁽" + diacritic + "⁾"
            if result:
                result[-1] += suffix
            else:
                result.append(suffix)
            continue

        if ch == "͡":
            if result:
                result[-1] += ch
            else:
                result.append(ch)
            i += 1
            if i < n:
                result[-1] += phonemes[i]
                i += 1
                while i < n and is_diacritic(phonemes[i]) and phonemes[i] != "͡":
                    result[-1] += phonemes[i]
                    i += 1
            continue

        token = ch
        i += 1
        while i < n:
            if phonemes[i] == "͡":
                token += phonemes[i]
                i += 1
                if i < n:
                    token += phonemes[i]
                    i += 1
                    while i < n and is_diacritic(phonemes[i]) and phonemes[i] != "͡":
                        token += phonemes[i]
                        i += 1
                else:
                    break
            elif is_diacritic(phonemes[i]):
                token += phonemes[i]
                i += 1
            else:
                break
        result.append(token)

    return result


def merge_phoneme_tokens(tokens: list[str]) -> list[str]:
    result: list[str] = []
    i = 0
    n = len(tokens)

    while i < n:
        t = tokens[i]

        if i == 0 and t.startswith("(") and t.endswith(")") and i + 1 < n:
            result.append(t + tokens[i + 1])
            i += 2
            continue

        if t in ("ˌ", "ˈ") and i + 1 < n:
            result.append(t + tokens[i + 1])
            i += 2
            continue

        if t == "ː" and result:
            result[-1] += "ː"
            i += 1
            continue

        if t.startswith("(") and t.endswith(")") and result:
            result[-1] += t
            i += 1
            continue

        result.append(t)
        i += 1

    return result


def process_word(phonemes: str) -> list[str]:
    return merge_phoneme_tokens(split_into_phonemes(phonemes))


def process_text(phonemes: list[str]) -> list[str]:
    result: list[str] = []
    for phoneme in phonemes:
        if phoneme.strip() == "" or all(ch in _PUNCTUATION for ch in phoneme):
            result.append(phoneme)
        else:
            result.extend(process_word(phoneme))
    return result
