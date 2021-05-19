# -*- coding: utf-8 -*-

import re
import string

import unidecode
import numpy as np
from nltk import ngrams
import random

MAXLEN = 34
NGRAM = 5
eng_alphabet = "abcdefghijklmnopqrstuvwxyz"
viet_alphabet = "áàảãạâấầẩẫậăắằẳẵặóòỏõọôốồổỗộơớờởỡợéèẻẽẹêếềểễệúùủũụưứừửữựíìỉĩịýỳỷỹỵđ"
digits = "0123456789"
punctuations = " _!\"\',\-\.:;?_\(\)\x00"

pattern = "^[" + "".join((eng_alphabet, viet_alphabet, digits, punctuations)) + "]+$"

accented_chars_vietnamese = [
    'á', 'à', 'ả', 'ã', 'ạ', 'â', 'ấ', 'ầ', 'ẩ', 'ẫ', 'ậ', 'ă', 'ắ', 'ằ', 'ẳ', 'ẵ', 'ặ',
    'ó', 'ò', 'ỏ', 'õ', 'ọ', 'ô', 'ố', 'ồ', 'ổ', 'ỗ', 'ộ', 'ơ', 'ớ', 'ờ', 'ở', 'ỡ', 'ợ',
    'é', 'è', 'ẻ', 'ẽ', 'ẹ', 'ê', 'ế', 'ề', 'ể', 'ễ', 'ệ',
    'ú', 'ù', 'ủ', 'ũ', 'ụ', 'ư', 'ứ', 'ừ', 'ử', 'ữ', 'ự',
    'í', 'ì', 'ỉ', 'ĩ', 'ị',
    'ý', 'ỳ', 'ỷ', 'ỹ', 'ỵ',
    'đ',
]
accented_chars_vietnamese.extend([c.upper() for c in accented_chars_vietnamese])
alphabet = list(('\x00 _' + string.ascii_letters + string.digits
                 + ''.join(accented_chars_vietnamese) + string.punctuation))


def remove_accent(text):
    return unidecode.unidecode(text)

def unide(txt):
    ran = random.sample(range(len(txt)),2 if 2 <len(txt) else 1)
    x, y = min(ran), max(ran)
    return txt[:x] + unidecode.unidecode(txt[x:y]) + txt[y:]

def make_noise(text):
    case = random.randrange(0,4)
    if (case == 0):
        text = unide(text)
    res = [i for i in text]
    if (case == 1):
        num_noise = random.randrange(0,3)
        size = range(len(text))
        noise = random.sample(size, num_noise)
        for idx in noise:
            place = alphabet.index(text[idx])
            padding = int(random.random()*8) if(random.random() < 0.8) else int(random.random()*20)
            l_o_r = random.randrange(0,2)
            sub = place - padding
            add = place + padding
            if (sub <= 0):
                place += padding
                l_o_r = 2
            if (add >= len(alphabet)):
                place -= padding
                l_o_r = 2
            if (l_o_r == 0):
                place -= padding
            if (l_o_r == 1) :
                place += padding
            res[idx] = alphabet[place]
    if (case == 2):
        num_add = random.randrange(0,3)
        add_noise = random.sample(range(len(text)), num_add)
        for id in add_noise:
            nse = random.sample(alphabet, int(random.random()+1))
            nse = ''.join(nse)
            res.insert(id, nse)
    if (case == 3):
        num_sub = random.randrange(0,3)
        sub_noise = random.sample(range(len(text)), num_sub)
        for id in sub_noise:
            if (id >= len(res)):
                continue
            res.pop(id)
    return ''.join(res)

def extract_phrases(text):
    return re.findall(r'\w[\w ]+', text)


def gen_ngrams(words, n=NGRAM):
    return ngrams(words.split(), n)


def encode(text, maxlen=MAXLEN):
    text = "\x00" + text
    x = np.zeros((maxlen, len(alphabet)))
    for i, c in enumerate(text[:maxlen]):
        x[i, alphabet.index(c)] = 1
    if i < maxlen - 1:
        for j in range(i + 1, maxlen):
            x[j, 0] = 1
    return x


def decode(x, calc_argmax=True):
    if calc_argmax:
        x = x.argmax(axis=-1)
    return ''.join(alphabet[i] for i in x)


def generate_data(data, batch_size=128):
    cur_index = 0
    while True:
        x, y = [], []
        for i in range(batch_size):
            y.append(encode(data[cur_index]))
            x.append(encode(make_noise(data[cur_index])))
            cur_index += 1

            if cur_index > len(data) - 1:
                cur_index = 0

        yield np.array(x), np.array(y)

