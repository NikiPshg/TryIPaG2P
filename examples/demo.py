from tryiparu import G2PModel

g2p = G2PModel(load_dataset=True)

if __name__ == "__main__":
    print(g2p(
    """
    текст в фонемы ипа формата
    """
    )) 

# ['tʲ', 'e', 'k', 's', 't', ' ', 'v', ' ', 'f', 'ɐ', 'n', 'ɛ', 'm', 'ɨ', ' ', 'ɪ', 'p', 'a', ' ', 'f', 'ɐ', 'r', 'm', 'a', 't', 'ə']