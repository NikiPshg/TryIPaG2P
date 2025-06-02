import unicodedata
import string

def is_diacritic(char):

    if unicodedata.combining(char) != 0:
        return True
    if char in "ʲː":
        return True
    return False

def split_into_phonemes(phonemes: str):
    phonemes_list = []
    i = 0
    n = len(phonemes)
    while i < n:
        if phonemes[i].isspace():
            i += 1
            continue

        # ( ː )
        if phonemes[i] == '(':
            group = '('
            i += 1  
            while i < n and phonemes[i] != ')':
                if not phonemes[i].isspace():
                    group += phonemes[i]
                i += 1
            if i < n and phonemes[i] == ')':
                group += ')'
                i += 1
            phonemes_list.append(group)
            continue

        # (⁽ ... ⁾)
        if phonemes[i] == "⁽":
            diacritic = ""
            i += 1  # пропускаем ⁽
            while i < n and phonemes[i] != "⁾":
                if not phonemes[i].isspace():
                    diacritic += phonemes[i]
                i += 1
            if i < n and phonemes[i] == "⁾":
                i += 1
            if phonemes_list:
                phonemes_list[-1] += "⁽" + diacritic + "⁾"
            else:
                phonemes_list.append("⁽" + diacritic + "⁾")
            continue

        if phonemes[i] == "͡":
            if phonemes_list:
                phonemes_list[-1] += phonemes[i]
            else:
                phonemes_list.append(phonemes[i])
            i += 1
            if i < n:
                phonemes_list[-1] += phonemes[i]
                i += 1
                while i < n and is_diacritic(phonemes[i]) and phonemes[i] != "͡":
                    phonemes_list[-1] += phonemes[i]
                    i += 1
            continue

        token = phonemes[i]
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
        phonemes_list.append(token)
    return phonemes_list

def merge_phoneme_tokens(tokens):
    result = []
    i = 0
    while i < len(tokens):
        t = tokens[i]

        if i == 0 and t.startswith("(") and t.endswith(")") and i+1 < len(tokens):
            result.append(t + tokens[i+1])
            i += 2
            continue

        if t in ["ˌ", "ˈ"] and i+1 < len(tokens):
            result.append(t + tokens[i+1])
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


def process_word(phonemes: str):
    tokens = split_into_phonemes(phonemes)
    merged_phonemes = merge_phoneme_tokens(tokens)
    return merged_phonemes

def process_text(phonemes: str):
    result = []
    for phoneme in phonemes:
        if phoneme.strip() == "" or all(ch in string.punctuation for ch in phoneme):  
            result.append(phoneme)
        else:
            result.extend(process_word(phoneme))
    return result


