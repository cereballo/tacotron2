from dp.phonemizer import Phonemizer


class TextProcessor:
    """
    Converts text to a list of symbols ids.
    """

    _text_symbols = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZäöüÄÖÜß")
    _phone_symbols = [
        'a', 'b', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
        'p', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'æ', 'ç', 'ð', 'ø',
        'ŋ', 'œ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɝ', 'ɹ', 'ɡ', 'ɪ', 'ʁ', 'ʃ', 'ʊ',
        'ʌ', 'ʏ', 'ʒ', 'ʔ', 'ˈ', 'ˌ', 'ː', '̃', '̍', '̥', '̩', '̯', '͡', 'θ'
    ]
    _replace_with_space = list('-')
    _punctuation = list('!\'(),.:;? ')
    _phonmizer: Phonemizer
    _symbol_id_map: dict[str, int]
    symbols: list[str]

    def __init__(self, phonemizer_path: str):
        self.phonemizer = Phonemizer.from_checkpoint(phonemizer_path)
        self.n_symbols = len(self._text_symbols) + len(self._phone_symbols) + len(self._punctuation)
        self.symbols = self._text_symbols + self._phone_symbols + self._punctuation
        self._symbol_id_map = {s: i for i, s in enumerate(self.symbols)}

    def __call__(self, text: str) -> list[int]:
        phones = self.phonemizer(text, lang="en_us")
        for s in self._replace_with_space:
            phones = phones.replace(s, " ")
        phones = [self._symbol_id_map[s] for s in phones]
        return phones
