"""
Test the extracted TorchScript-based Silero TE model.
This demonstrates loading via torch.jit.load() instead of torch.package.
Uses correct helper functions from the original release_module.py.
"""
import sys
from pathlib import Path
from typing import List

# Fix encoding for Windows console
sys.stdout.reconfigure(encoding='utf-8')

import torch
import numpy as np

MODEL_DIR = Path(__file__).parent / "models" / "silero-te"

print(f"Model directory: {MODEL_DIR}")
print(f"PyTorch version: {torch.__version__}")
print("=" * 60)


# ============================================================================
# Helper functions from release_module.py (EXACT COPY)
# ============================================================================

def is_transformed_char(char: str):
    return char[0] == '{' and char[-1] == '}' and char[1:-1].isdigit()


def split_into_chars(text: str):
    if is_transformed_char(text):
        return [text]

    splitted = []
    char = ""
    uni_start = False

    for c in text:
        if not uni_start:
            if c != '{':
                splitted.append(c)
            else:
                char += c
                uni_start = True
        elif uni_start:
            if c.isdigit():
                char += c
            elif c == '}':
                char += c
                splitted.append(char)
                char = ""
                uni_start = False
            else:
                assert '{' in text, "Service character { in text"

    return splitted


def unitoken_into_token(unitoken: str):
    chars = split_into_chars(unitoken)
    token = ""

    for c in chars:
        if is_transformed_char(c):
            token += chr(int(c[1:-1]))
        else:
            token += c

    return token


def process_unicode(text, uni_symbols):
    """Process text to handle unicode characters."""
    processed = ""

    for c in text:
        if ord(c) < 127:
            processed += c
        elif c not in uni_symbols:
            processed += '&'
        else:
            processed += '{' + str(ord(c)) + '}'

    return processed


def enhance_tokens(tokens, punct, capital, index2punct):
    """Apply punctuation and capitalization to tokens."""
    output = []
    punct, capital = punct.cpu().numpy(), capital.cpu().numpy()

    sentence_end = False

    for token, p, c in zip(tokens, punct, capital):
        if sentence_end:
            if c == 0:
                c = 1
            sentence_end = False

        if c == 1:
            if token[0].isalnum():
                token = token[0].upper() + token[1:]
            else:
                if len(token) < 2:
                    print(token)
                token = '##' + token[2].upper() + token[3:]
        if c == 2:
            token = token.upper()
        output.append(token)

        if p:
            symbol = index2punct[p]
            output.append(symbol)

            if symbol in '.!?':
                sentence_end = True

    return output


# ============================================================================
# TeModelJit class
# ============================================================================

class TeModelJit:
    """
    TorchScript-based Silero Text Enhancement model.
    Uses torch.jit.load() instead of torch.package.PackageImporter.
    This should work in PyInstaller frozen builds.
    """

    def __init__(self, model_dir=None, pad=True):
        """
        Args:
            model_dir: Path to directory with te_model_jit.pt and te_tokenizer_jit.pt
            pad: Whether to pad input tokens (required for correct end punctuation)
        """
        if model_dir is None:
            model_dir = MODEL_DIR

        self.model_dir = Path(model_dir)
        self.pad = pad  # IMPORTANT: pad=True is required for correct punctuation!
        self.device = torch.device('cpu')

        # Load TorchScript models
        print(f"Loading TorchScript models from {self.model_dir}...")

        model_path = self.model_dir / "te_model_jit.pt"
        tokenizer_path = self.model_dir / "te_tokenizer_jit.pt"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

        torch.set_grad_enabled(False)

        self.model = torch.jit.load(str(model_path), map_location='cpu')
        self.model.eval()

        self.tokenizer = torch.jit.load(str(tokenizer_path), map_location='cpu')

        print(f"  Model type: {type(self.model).__name__}")
        print(f"  Tokenizer type: {type(self.tokenizer).__name__}")

        # Get uni_symbols from tokenizer
        self.uni_symbols = set()
        for unitoken in self.tokenizer.uni_vocab:
            self.uni_symbols.update(set(unitoken_into_token(unitoken)))

        self.index2punct = {1: '.', 2: ',', 3: '-', 4: '!', 5: '?', 6: '_'}

        self.examples = [
            'afterwards we were taken to one of the undamaged dormitory buildings',
            'der fruhling kam spät und war ungewöhnlich regnerisch',
            'ель многолетнее растение распространенное в евразии и северной америке',
            'cómo dar una definición de su identidad'
        ]

        print("Model loaded successfully!")

    def pad_ids(self, ids, limit=18):
        """Pad token IDs for batch processing."""
        if self.pad:
            if ids.shape[1] < limit:
                ids_padded = torch.LongTensor(1, limit)
            else:
                ids_padded = torch.LongTensor(1, min(ids.shape[1] + limit, 512))

            ids_padded.zero_()
            ids_padded[0, :ids.shape[1] - 1] = ids[0, :-1]
            ids_padded[0, -1] = ids[0, -1]

            att_mask = torch.ones_like(ids_padded)
            att_mask[0, ids.shape[1] - 1:-1].zero_()

            return ids_padded, True, att_mask
        else:
            return ids, False, torch.ones_like(ids)

    def enhance_textblock(self, text, lan_id):
        """Enhance a single text block."""
        device = self.device
        lan_id = lan_id.to(device)

        with torch.no_grad():
            x = process_unicode(text, self.uni_symbols)
            x = torch.tensor([self.tokenizer.convert_string_to_ids(x)])
            x, pad, att_mask = self.pad_ids(x)
            punct, capital = self.model(x.to(device), att_mask.to(device), lan_id)
            punct_argmax = torch.argmax(punct, dim=-1)
            capital_argmax = torch.argmax(capital, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens([item for item in x[0]])
        tokens = list(map(unitoken_into_token, tokens))

        if pad:
            pad_token = self.tokenizer.pad_token
            tokens = tokens[:tokens.index(pad_token)] + tokens[-1:]
            punct_argmax = punct_argmax[:, att_mask[0].bool()]
            capital_argmax = capital_argmax[:, att_mask[0].bool()]

        enhanced_tokens = enhance_tokens(
            tokens[1:-1],
            punct_argmax[0][1:-1],
            capital_argmax[0][1:-1],
            self.index2punct
        )

        return self.tokenizer.convert_tokens_to_string(enhanced_tokens)

    def count_occurrences(self, text, char):
        if char in text:
            counter = 0
            ind = -1
            while True:
                ind = text.find(char, ind + 1)
                if ind == -1:
                    break
                counter += 1
            return counter
        return 0

    def enhance_long_textblock(self, text, lan_id, len_limit):
        """Enhance a long text block by splitting into chunks."""
        result = ''
        words = text.split()

        _from, _to = 0, 0
        while _to < len(words):
            _to = _from + len_limit
            block = ' '.join(words[_from:_to])

            enhanced = self.enhance_textblock(block, lan_id)
            enhanced_words = enhanced.split()
            symbols = ''.join([word[-1] for word in enhanced_words])

            ind = max(symbols.rfind('.'), symbols.rfind('!'), symbols.rfind('?')) + 1
            ind += self.count_occurrences(enhanced, '-')

            result += ' '.join(enhanced_words[:ind]) + ' '
            _from = _from + ind

        return result

    def enhance_text(self, text, lan='en', len_limit=150):
        """
        Enhance text by adding punctuation and capitalization.

        Args:
            text: Input text without punctuation
            lan: Language code ('en', 'de', 'es', 'ru')
            len_limit: Maximum words per block

        Returns:
            Enhanced text with punctuation and capitalization
        """
        lan2index = {'en': 0, 'de': 1, 'es': 2, 'ru': 3}
        if lan not in lan2index:
            lan = 'en'
        lan_id = torch.tensor([[[lan2index[lan]]]])

        if len(text.split()) < len_limit:
            enhanced = self.enhance_textblock(text, lan_id)
        else:
            enhanced = self.enhance_long_textblock(text, lan_id, len_limit)

        # Spanish inverted punctuation
        if lan == 'es' and ('?' in enhanced or '!' in enhanced):
            for m, rm in zip('?!', '¿¡'):
                ind = 0
                prev = 0
                while ind < len(enhanced) and m in enhanced[ind:]:
                    ind = enhanced.find(m, ind)
                    if ind == -1:
                        break
                    part = enhanced[prev:ind]
                    end = max(part.rfind('.'), part.rfind('!'), part.rfind('?'))
                    if end != -1:
                        prev += end + 1
                        enhanced = enhanced[:prev + 1] + rm + enhanced[prev + 1:]
                    else:
                        enhanced = rm + enhanced
                    ind += 2

        enhanced = enhanced.replace('_', ' —').strip()
        return enhanced


# ============================================================================
# Test
# ============================================================================

print("\n1. Creating TeModelJit instance...")
model = TeModelJit()

print("\n2. Testing enhance_text():")
print("-" * 40)

test_cases = [
    ("привет как дела", "ru"),
    ("hello how are you", "en"),
    ("der fruhling kam spät", "de"),
    ("cómo estás hoy", "es"),
    ("я иду домой сегодня вечером", "ru"),
    ("this is a test of the text enhancement model it should add punctuation", "en"),
]

for text, lang in test_cases:
    try:
        result = model.enhance_text(text, lang)
        print(f"  [{lang}] '{text}'")
        print(f"       -> '{result}'")
    except Exception as e:
        print(f"  [{lang}] '{text}' -> ERROR: {e}")
        import traceback
        traceback.print_exc()

print("\n3. Comparing with original model:")
print("-" * 40)

# Load original for comparison
from torch.package import PackageImporter
original_path = MODEL_DIR / "v2_4lang_q.pt"
if original_path.exists():
    importer = PackageImporter(str(original_path))
    original_model = importer.load_pickle("te_model", "model")

    all_match = True
    for text, lang in test_cases:
        orig_result = original_model.enhance_text(text, lang)
        new_result = model.enhance_text(text, lang)

        match = orig_result == new_result
        all_match = all_match and match
        status = "MATCH" if match else "DIFFER"
        print(f"  [{lang}] '{text}'")
        print(f"       Original: '{orig_result}'")
        print(f"       JIT:      '{new_result}'")
        print(f"       Status:   {status}")
        print()

    if all_match:
        print("ALL TESTS PASSED - JIT model produces identical results!")
    else:
        print("SOME TESTS DIFFER - investigation needed")

print("\n" + "=" * 60)
print("TEST COMPLETE")
