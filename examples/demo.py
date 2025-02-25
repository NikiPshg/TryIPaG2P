import sys

src_path = ''

if not (src_path in sys.path ):
    sys.path.append(src_path)

from g2p.g2p import G2PModel

g2p = G2PModel()

print(g2p("Солнце,           Сердце .Поздно :Чувство  ?? Здравствуйте ! Лестница 1 Радостный 2 Местный @ Праздник Честный")) 
#'sont͡sɨ', ',',
# 'sʲɪrt͡sɛ', '.',
# 'poznə', ':',
# 't͡ɕustvə',  '?', '?',
# 'zdrastvʊjtʲe', '!',
#  'lʲesʲnʲɪt͡sə', ' ',
#  'ɛmtɛ', ' ',
#  'radəsnɨj', ' ',
#  'ɛt͡sʲɪmʲɪ', ' ',
#  'mʲesnɨj', '@',
#  'prazʲnʲɪk', ' ',
#  't͡ɕesnɨj']
