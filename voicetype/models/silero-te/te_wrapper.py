import os
import torch
from typing import List


class TeModel:
    def __init__(self, model_dir, pad=False):
        self.model, self.tokenizer = self.init_jit_model(model_dir)
        self.pad = pad

        self.uni_symbols = set()
        for unitoken in self.tokenizer.uni_vocab:
            self.uni_symbols.update(set(unitoken_into_token(unitoken)))

        self.index2punct = {1: '.', 2: ',', 3: '-', 4: '!', 5: '?', 6: '_'}
        self.device = torch.device('cpu')

        self.examples = ['afterwards we were taken to one of the undamaged dormitory buildings',
                         'der fruhling kam spät und war ungewöhnlich regnerisch',
                         'ель многолетнее растение распространенное в евразии и северной америке',
                         'cómo dar una definición de su identidad']

    def init_jit_model(self, model_dir: str):
        torch.set_grad_enabled(False)

        model_path = os.path.join(model_dir, 'model')
        tokenizer_path = os.path.join(model_dir, 'tokenizer')
        model = torch.jit.load(model_path, map_location='cpu')
        model.eval()

        tokenizer = torch.jit.load(tokenizer_path, map_location='cpu')
        return model, tokenizer

    def pad_ids(self, ids, limit=18):
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
        device = torch.device('cpu')
        lan_id = lan_id.to(device)

        with torch.no_grad():
            x = process_unicode(text, self.uni_symbols)
            x = torch.tensor([self.tokenizer.convert_string_to_ids(x)])
            x, pad, att_mask = self.pad_ids(x)
            punct, capital = self.model(x.to(device), att_mask.to(device), lan_id)
            punct = torch.argmax(punct, dim=-1)
            capital = torch.argmax(capital, dim=-1)

        tokens = self.tokenizer.convert_ids_to_tokens([item for item in x[0]])
        tokens = list(map(unitoken_into_token, tokens))

        if pad:
            tokens = tokens[:tokens.index(self.tokenizer.pad_token)] + tokens[-1:]
            punct = punct[:, att_mask[0].bool()]
            capital = capital[:, att_mask[0].bool()]

        return self.tokenizer.convert_tokens_to_string(enhance_tokens(tokens[1:-1],
                                                                      punct[0][1:-1],
                                                                      capital[0][1:-1],
                                                                      self.index2punct))

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
        lan2index = {'en': 0, 'de': 1, 'es': 2, 'ru': 3}
        if lan not in lan2index:
            lan = 'en'
        lan_id = torch.tensor([[[lan2index[lan]]]])

        if len(text.split()) < len_limit:
            enhanced = self.enhance_textblock(text, lan_id)
        else:
            enhanced = self.enhance_long_textblock(text, lan_id, len_limit)

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


def enhance_tokens(tokens, punct, capital, index2punct):
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


def is_transformed_char(char: str):
    return char[0] == '{' and char[-1] == '}' and char[1:-1].isdigit()


def split_into_chars(text: str):
    if is_transformed_char(text):
        return [text]

    splitted = torch.jit.annotate(List[str], [])
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
    processed = ""

    for c in text:
        if ord(c) < 127:
            processed += c
        elif c not in uni_symbols:
            processed += '&'
        else:
            processed += '{' + str(ord(c)) + '}'

    return processed
