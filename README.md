# TryIpaRu

Русский графемно-фонемный преобразователь (G2P) с выводом в формате IPA.

Использует Transformer encoder-decoder модель и словарь на ~400K слов для быстрого преобразования русского текста в фонемную транскрипцию.

## Установка

```bash
pip install git+https://github.com/NikiPshg/TryIPaG2P.git
```

Или для локальной разработки:

```bash
git clone https://github.com/NikiPshg/TryIPaG2P.git
cd TryIPaG2P
pip install -e .
```

## Быстрый старт

```python
from tryiparu import G2PModel

g2p = G2PModel()

result = g2p("текст в фонемы ипа формата")
print(result)
# ['tʲ', 'e', 'k', 's', 't', ' ', 'v', ' ', 'f', 'ɐ', 'n', 'ɛ', 'm', 'ɨ', ' ', 'ɪ', 'p', 'a', ' ', 'f', 'ɐ', 'r', 'm', 'a', 't', 'ə']
```

## Параметры `G2PModel`

| Параметр | Тип | По умолчанию | Описание |
|---|---|---|---|
| `device` | `str \| None` | `None` | Устройство (`"cpu"`, `"cuda"`). Автоопределение если `None` |
| `tokenizer_file` | `str \| None` | `None` | Путь к BPE tokenizer JSON |
| `model_weights` | `str \| None` | `None` | Путь к весам модели (.pt) |
| `load_dataset` | `bool` | `True` | Загрузить словарь для ускорения (lookup вместо inference) |

## Как это работает

1. Входной текст разбивается на токены (слова и пунктуация)
2. Каждое слово ищется в словаре (~400K записей)
3. Если слова нет в словаре — запускается Transformer для генерации фонем
4. Результат кешируется для повторных запросов
5. Пост-обработка объединяет диакритики, аффрикаты, ударения

## Зависимости

- `torch >= 2.0`
- `tokenizers >= 0.13`

## Лицензия

MIT
