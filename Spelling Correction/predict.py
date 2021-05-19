# -*- coding: utf-8 -*-
from collections import Counter

from keras.engine.saving import load_model

from helpers import *

NGRAM = 5
model = load_model('trained_model/model1_0.0832_0.9756.h5')


def extract_phrases(text):
    pattern = r'\w[\w ]*|\s\W+|\W+'
    return re.findall(pattern, text)


def guess(ngram):
    text = ' '.join(ngram)
    preds = model.predict(np.array([encode(text)]), verbose=0)
    return decode(preds[0], calc_argmax=True).strip('\x00')


def padding_word(text):
    len_text = len(text.split(" "))
    padding = " 0" * (10 - len_text)
    new_text = text + padding
    return new_text


def add_accent(text):
    new_text = padding_word(text)
    ngrams = list(gen_ngrams(new_text, n=NGRAM))
    guessed_ngrams = list(guess(ngram) for ngram in ngrams)
    candidates = [Counter() for _ in range(len(guessed_ngrams) + NGRAM - 1)]
    for nid, ngram in enumerate(guessed_ngrams):
        for wid, word in enumerate(re.split(' +', ngram)):
            candidates[nid + wid].update([word])
    print(candidates)
    tone_predict = ' '.join([max(dict(i)) for i in candidates if dict(i)])
    output = tone_predict[:len(text)]
    print(output)
    return output

def accent_sentence(sentence):
    list_phrases = extract_phrases(sentence)
    output = ""
    for phrases in list_phrases:
        if len(phrases.split()) < 2 or not re.match("\w[\w ]+", phrases):
            output += phrases
        else:
            output += add_accent(phrases)
            if phrases[-1] == " ":
                output += " "
    return output


with open("asset/test.txt") as f:
    content = f.read().split("\n")

# text = [accent_sentence(line) for line in content]
i = 0
text = []
# for line in content:
#     print(i)
#     i = i + 1
#     text.append(accent_sentence(line))
# with open("asset/accent_test.txt", "w") as f:
#     f.write("\n".join(i for i in text))

# text = "Tuy nhien, tren thuc te, moi nguoi deu hieu rang, viec cuoc cai to noi cac lan nay cua ba May la mot canh bac nham xac dinh va ap dat quyen luc lanh dao cua ba doi voi nhung thanh vien noi cac, trong do co nhung nguoi da the hien bat dong chinh kien voi ba trong van de Brexit va mot so van de khac ve chinh tri, kinh te, xa hoi."
# text = 'Đây sẽ là chiếm thắng hiển hách nhất của zân tộk'
# text = 'Ta đi trên con đường ngày mai. Dẫu phong ba hay hiểm nguy. Anh luôn bên em vượq qua. Đến mai nà chúng ta già. Vẫn bên nhau dựa lưng.'
text = 'Vi to quoc xa hoi chu nghia'
accent_text = accent_sentence(text)
print(accent_text)
