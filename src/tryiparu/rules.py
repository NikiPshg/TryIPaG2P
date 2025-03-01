import unicodedata
import string

def is_diacritic(char):

    if unicodedata.combining(char) != 0:
        return True
    if char in "ʲː":
        return True
    return False

def split_into_phonemes(word):
    phonemes = []
    i = 0
    n = len(word)
    while i < n:
        if word[i].isspace():
            i += 1
            continue

        # ( ː )
        if word[i] == '(':
            group = '('
            i += 1  
            while i < n and word[i] != ')':
                if not word[i].isspace():
                    group += word[i]
                i += 1
            if i < n and word[i] == ')':
                group += ')'
                i += 1
            phonemes.append(group)
            continue

        # (⁽ ... ⁾)
        if word[i] == "⁽":
            diacritic = ""
            i += 1  # пропускаем ⁽
            while i < n and word[i] != "⁾":
                if not word[i].isspace():
                    diacritic += word[i]
                i += 1
            if i < n and word[i] == "⁾":
                i += 1
            if phonemes:
                phonemes[-1] += "⁽" + diacritic + "⁾"
            else:
                phonemes.append("⁽" + diacritic + "⁾")
            continue

        if word[i] == "͡":
            if phonemes:
                phonemes[-1] += word[i]
            else:
                phonemes.append(word[i])
            i += 1
            if i < n:
                phonemes[-1] += word[i]
                i += 1
                while i < n and is_diacritic(word[i]) and word[i] != "͡":
                    phonemes[-1] += word[i]
                    i += 1
            continue

        token = word[i]
        i += 1
        while i < n:
            if word[i] == "͡":
                token += word[i]
                i += 1
                if i < n:
                    token += word[i]
                    i += 1
                    while i < n and is_diacritic(word[i]) and word[i] != "͡":
                        token += word[i]
                        i += 1
                else:
                    break
            elif is_diacritic(word[i]):
                token += word[i]
                i += 1
            else:
                break
        phonemes.append(token)
    return phonemes

def merge_phoneme_tokens(tokens):
    """
    "(j)" + "ɪ" → "(j)ɪ")
    """
    result = []
    i = 0
    while i < len(tokens):
        if i == 0 and tokens[i].startswith("(") and tokens[i].endswith(")") and i + 1 < len(tokens):
            merged = tokens[i] + tokens[i+1]
            result.append(merged)
            i += 2
        elif tokens[i] in ["ˌ", "ˈ"] and i + 1 < len(tokens):
            merged = tokens[i] + tokens[i+1]
            result.append(merged)
            i += 2
        elif tokens[i].startswith("(") and tokens[i].endswith(")") and result:
            result[-1] = result[-1] + tokens[i]
            i += 1
        else:
            result.append(tokens[i])
            i += 1
    return result

def process_word(word):
    tokens = split_into_phonemes(word)
    merged_tokens = merge_phoneme_tokens(tokens)
    return merged_tokens

def process_text(text_list):
    result = []
    for token in text_list:
        if token.strip() == "" or all(ch in string.punctuation for ch in token):  
            result.append(token)
        else:
            result.extend(process_word(token))
    return result


