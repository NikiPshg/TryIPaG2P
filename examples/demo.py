import sys

src_path = r'C:\Users\RedmiBook\Apython\TryIPaG2P\src'

if not (src_path in sys.path ):
    sys.path.append(src_path)

from g2p.g2p import G2PModel

g2p = G2PModel()
print(g2p(
"""
лесопромышленник, здраствуйте ,лестница . мтуси? 
"""
)) 
# ['lʲ', 'e', 's', 'ə', 'p', 'r', 'ɐ', 'm', 'ɨ', 'ʂ', 'lʲ', 'ɪ', 'nʲ(ː)', 'ɪ', 'k', ',',
# 'z', 'd', 'r', 'a', 's', 't', 'v', 'ʊ', 'j', 'tʲ', 'e', ',',
# 'lʲ', 'e', 'sʲ', 'nʲ', 'ɪ', 't͡s', 'ə', '.',
# 'm', 't', 'ʊ', 'sʲ', 'i', '?']