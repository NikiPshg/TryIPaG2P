from tryiparu import G2PModel

if __name__ == "__main__":
    g2p = G2PModel(load_dataset=True)

    result = g2p("текст в фонемы ипа формата")
    print(result)
    # ['tʲ', 'e', 'k', 's', 't', ' ', 'v', ' ', 'f', 'ɐ', 'n', 'ɛ', 'm', 'ɨ', ' ', 'ɪ', 'p', 'a', ' ', 'f', 'ɐ', 'r', 'm', 'a', 't', 'ə']
