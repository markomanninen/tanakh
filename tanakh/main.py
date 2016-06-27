#!/usr/local/bin/python
# -*- coding: utf-8 -*-
# file: main.py

import re
import difflib
import numpy as np
import pandas as pd
from os.path import isfile
from IPython.display import HTML
from .strongs3.abnum.remarkuple import helper as h
from .strongs3 import hebrew as hbr
from .strongs3.abnum import find_cumulative_indices, Abnum, hebrew
from .strongs3.abnum.romanize.heb import letters

KtavIvri_Mapping = (
    [1, 'alef'], [2, 'beth'], [3, 'gimel'], 
    [3, 'shin'], [4, 'daleth'], [4, 'tau'],
    [5, 'he'], [6, 'vau'], [7, 'zayin'], 
    [8, 'heth'], [9, 'teth'], [10, 'yod'], 
    [20, 'kaph'], [30, 'lamed'], [40, 'mem'], 
    [50, 'nun'], [60, 'samekh'], [70, 'ayin'], 
    [80, 'pe'], [90, 'tsade'], [100, 'qoph'], 
    [200, 'resh'], [20, 'final_kaph'], [40, 'final_mem'], 
    [50, 'final_nun'], [80, 'final_pe'], [90, 'final_tsade']
)

h = Abnum(hebrew, KtavIvri_Mapping)

# should be gematria!
isopsephy = h.value
to_roman = h.convert
preprocess = h.preprocess

from diff import diff_match_patch

diff = diff_match_patch()
find = hbr.find

__version__ = "0.0.1"

booknames = {
    "01O": "Genesis",
    "02O": "Exodus",
    "03O": "Leviticus",
    "04O": "Numbers",
    "05O": "Deuteronomy",
    "06O": "Joshua",
    "07O": "Judges",
    "08O": "Ruth",
    "09O": "1 Samuel",
    "10O": "2 Samuel",
    "11O": "1 Kings",
    "12O": "2 Kings",
    "13O": "1 Chronicles",
    "14O": "2 Chronicles",
    "15O": "Ezra",
    "16O": "Nehemiah",
    "17O": "Esther",
    "18O": "Job",
    "19O": "Psalms",
    "20O": "Proverbs",
    "21O": "Ecclesiastes",
    "22O": "Song of Solomon",
    "23O": "Isaiah",
    "24O": "Jeremiah",
    "25O": "Lamentations",
    "26O": "Ezekiel",
    "27O": "Daniel",
    "28O": "Hosea",
    "29O": "Joel",
    "30O": "Amos",
    "31O": "Obadiah",
    "32O": "Jonah",
    "33O": "Micah",
    "34O": "Nahum",
    "35O": "Habakkuk",
    "36O": "Zephaniah",
    "37O": "Haggai",
    "38O": "Zechariah",
    "39O": "Malachi",
    "40N": "Matthew",
    "41N": "Mark",
    "42N": "Luke",
    "43N": "John",
    "44N": "Acts of the Apostles",
    "45N": "Romans",
    "46N": "1 Corinthians",
    "47N": "2 Corinthians",
    "48N": "Galatians",
    "49N": "Ephesians",
    "50N": "Philippians",
    "51N": "Colossians",
    "52N": "1 Thessalonians",
    "53N": "2 Thessalonians",
    "54N": "1 Timothy",
    "55N": "2 Timothy",
    "56N": "Titus",
    "57N": "Philemon",
    "58N": "Hebrews",
    "59N": "James",
    "60N": "1 Peter",
    "61N": "2 Peter",
    "62N": "1 John",
    "63N": "2 John",
    "64N": "3 John",
    "65N": "Jude",
    "66N": "Revelation"
}

manuscripts = {
    "hebrew_modern": {
        "variations": {"var0": "Hebrew: Modern"},
        "source": "hebrew_modern_utf8"},
    "aleppo": {
        "variations": {"var0": "Hebrew OT: Aleppo Codex"},
        "source": "aleppo_utf8"},
    "wlc_consonants": {
        "variations": {"var0": "Hebrew OT: Westminster Leningrad Codex"},
        "source": "wlc_consonants_utf8"},
    "hebrew_bhs_consonants": {
        "variations": {"var0": "Hebrew OT: BHS"},
        "source": "hebrew_bhs_consonants_utf8"}
}

# will be transformed to pandas DataFrame
manuscript_vocabulary = {k: {'var0': {}, 'var1': {}, 'var2': {}} for k in manuscripts.keys()}

sletters = ''.join(letters)

c = '([%s]+) ([^%s]+)' % (sletters, sletters)
regex_word_strong_morph = re.compile(c)

c = '([%s]+)' % (sletters)
regex_word_strong_morph2 = re.compile(c)

c = '([%s]+)' % sletters
regex_word_isopsephy = re.compile(c)

c = '{VAR1: ([%s0-9A-Z\- ]+)}' % sletters

regex_variation1 = re.compile(c)

c = '{VAR2: ([%s0-9A-Z\- ]+)}' % sletters

regex_variation2 = re.compile(c)

regex_word_strong_morph_brackets = re.compile('\[(.*)\]')

manuscript_original_dir = "data_original/%s/%s"
manuscript_processed_dir = "data_processed/%s/%s"

#letters = "αΑβΒγΓδΔεΕϛϚϜϝζΖηΗθΘιΙυϒYκΚϡϠͲͳλΛωΩμΜτΤνΝξΞοΟσΣϹϲςπΠχΧϙϘϞϟρΡψΨφΦ"
#c = '([%s]+) ([^%s]+)' % (letters, letters)
#c = "([αΑβΒγΓδΔεΕϛϚϜϝζΖηΗθΘιΙυϒYκΚϡϠͲͳλΛωΩμΜτΤνΝξΞοΟσΣϹϲςπΠχΧϙϘϞϟρΡψΨφΦ]+.*?.-.{3})"
#c = u'([Ͱ-ϡ]+) ([A-Z0-9-]+)(?: ([A-Z0-9-]+))? ([A-Z0-9-]+)(?=\\s|$)'

def load_dataframe(filename, manuscript, var = None):
    global manuscript_vocabulary

    csvOriginalFileName = manuscript_original_dir % (manuscript, filename)

    if var and var != 'var0':
        filename += "_%s" % var
    else:
        var = 'var0'

    csvProcessedFileName = manuscript_processed_dir % (manuscript, filename)

    if isfile(csvProcessedFileName + ".csv"):
        print ("Retrieving data from local csv copy (%s) ..." % csvProcessedFileName)
        manuscript_vocabulary[manuscript][var] = pd.read_csv(csvProcessedFileName + "_dict.csv")
        df = pd.read_csv(csvProcessedFileName + ".csv")
        df['text'] = df['text'].fillna('')
        df['text_isopsephy'] = df['text'].apply(lambda verse: verse_isopsephy_numbers(verse))
        df['total_isopsephy'] = df['text_isopsephy'].apply(lambda x: sum(x))
        return df

    print ("Processing data from original csv file (%s) ..." % (csvOriginalFileName))

    df = pd.read_csv(csvOriginalFileName + ".csv", sep="	", index_col=False)

    if 'orig_subverse' in df:
        del df['orig_subverse']
    if 'order_by' in df:
        del df['order_by']

    df['orig_book_index'] = df['orig_book_index'].apply(lambda index: booknames[index])
    df['text'] = df['text'].apply(lambda verse: parse_verse(verse, manuscript, manuscript_vocabulary[manuscript][var], var))
    #df['text'] = df['text'].fillna('missing')
    df.to_csv(csvProcessedFileName + ".csv", index=False)

    df['text_isopsephy'] = df['text'].apply(lambda verse: verse_isopsephy_numbers(verse))
    df['total_isopsephy'] = df['text_isopsephy'].apply(lambda x: sum(x))

    manuscript_vocabulary[manuscript][var] = pd.DataFrame(manuscript_vocabulary[manuscript][var])
    manuscript_vocabulary[manuscript][var].to_csv(csvProcessedFileName + "_dict.csv", index=False)

    return df

#v = "βιβλος G976 N-NSF γενεσεως G1078 N-GSF ιησου G2424 N-GSM χριστου G5547 N-GSM υιου G5207 N-GSM δαβιδ G1138 N-PRI υιου G5207 N-GSM αβρααμ G11 N-PRI"
#print pd.DataFrame(parse_verse(v))

# [] are found 14 times on manuscripts and are removed on this application
#v = "μονω G3441 A-DSM σοφω G4680 A-DSM θεω G2316 N-DSM δια G1223 PREP ιησου G2424 N-GSM χριστου G5547 N-GSM  {VAR1: ω G3739 R-DSM } η G3588 T-NSF δοξα G1391 N-NSF εις G1519 PREP τους G3588 T-APM αιωνας G165 N-APM αμην G281 HEB [προς G4314 PREP ρωμαιους G4514 A-APM εγραφη G1125 G5648 V-2API-3S απο G575 PREP κορινθου G2882 N-GSF δια G1223 PREP φοιβης G5402 N-GSF της G3588 T-GSF διακονου G1249 N-GSF της G3588 T-GSF εν G1722 PREP κεγχρεαις G2747 N-DPF εκκλησιας G1577 N-GSF]"
#print pd.DataFrame(parse_verse(v))

# {VAR1: } and {VAR2: } are found few hundred times on manuscripts. VAR2 is removed and VAR1 is kept.
#v = "και G2532 CONJ υμας G5209 P-2AP νεκρους G3498 A-APM οντας G5607 G5752 V-PXP-APM εν G1722 PREP τοις G3588 T-DPN παραπτωμασιν G3900 N-DPN και G2532 CONJ τη G3588 T-DSF ακροβυστια G203 N-DSF της G3588 T-GSF σαρκος G4561 N-GSF υμων G5216 P-2GP  {VAR1: συνεζωποιησεν G4806 V-AAI-3S } {VAR2: συνεζωοποιησεν G5656 V-AAI-3S } συν G4862 PREP αυτω G846 P-DSM χαρισαμενος G5483 G5666 V-ADP-NSM  {VAR1: ημιν G2254 P-1DP } {VAR2: υμιν G5213 P-2DP } παντα G3956 A-APN τα G3588 T-APN παραπτωματα G3900 N-APN"
#print pd.DataFrame(parse_verse(v))

def parse_verse(verse, manuscript, vocabulary, var = "var0"):
    verse = str(verse)
    # variation 1 marked with {VAR1 ...}
    x = regex_variation1.findall(verse)
    if x:
        for y in x:
            y = y.replace('[', '').replace(']', '')
            verse = verse.replace('{VAR1: '+y+'}', (y if var in ['var0', 'var1'] else '')).replace('  ', ' ')
    # variation 2 marked with {VAR2 ... }
    x = regex_variation2.findall(verse)
    if x:
        for y in x:
            y = y.replace('[', '').replace(']', '')
            verse = verse.replace('{VAR2: '+y+'}', (y if var == 'var2' else '')).replace('  ', ' ')
    # uncertain parts marked between [] are removed
    x = regex_word_strong_morph_brackets.search(verse)
    if x:
        verse = verse.replace('['+x.group(1)+']', '')
    # collecting and returning words from verse
    verse_words = []
    if manuscript == 'hebrew_modern' or \
       manuscript == 'hebrew_modern' or \
       manuscript == 'wlc_consonants' or \
       manuscript == 'hebrew_bhs_consonants':
        r = regex_word_strong_morph2
        verse = preprocess(verse)
        # aleppo contains first syllable as a verse number indicator 
        # which needs to be removed from processed content
        start = True
        if manuscript == 'aleppo':
            start = False
        for tupleitem in [[x] for x in r.findall(verse)]:
            if not start:
                start = True
                continue
            item = []
            # init word
            item.append(tupleitem[0])
            # add word for verse
            verse_words.append(item[0])
            # if item is not found from vocabulary, then add it
            if item[0] not in vocabulary:
                # see, if we can find lemma from Strong's dictionary
                strong = "".join([x for x in find(r"(?i)^%s$" % item[0], 'word').lemma][:1])
                item.append(strong)
                # we don't know exact morph...
                item.append("")
                # isopsephy
                item.append(isopsephy(item[0]))
                # adding transliteration
                #trans = trans.decode()
                item.append(to_roman(item[0]))
                # adding item to dictionary, but removing first item (word) from tuple
                vocabulary[item[0]] = item[1:]
    else:
        r = regex_word_strong_morph
        for tupleitem in [[x[0]] + x[1].split() for x in regex_word_strong_morph.findall(verse)]:
            item = []
            # word
            item.append(tupleitem[0])
            # lemma
            item.append(tupleitem[1])
            # morph
            if len(tupleitem) == 4:
                item.append(tupleitem[3])
            elif len(tupleitem) == 3:
                item.append(tupleitem[2])
            else:
                item.append("")
            # isopsephy
            item.append(isopsephy(item[0]))
            # adding transliteration
            item.append(to_roman(item[0]))
            # add word for verse
            verse_words.append(item[0])
            # adding item to dictionary, but removing first item (greek word) from tuple
            vocabulary[item[0]] = item[1:]

    return " ".join(verse_words)

def verse_isopsephy_numbers(verse):
    return list(map(isopsephy, verse.split())) if type(verse) is str else []

def match_isopsephy_combinations(num, dataframe):
    dataframe['search'] = dataframe['text_isopsephy'].apply(lambda numbers: find_cumulative_indices(numbers, num))
    rows = dataframe[dataframe['search'] != ''].sort('orig_book_index', ascending=True)
    del dataframe['search']
    return rows.sort(['orig_book_index', 'orig_chapter', 'orig_verse'])

def match_phrase(phrase, dataframe, books = []):
    dataframe['text'] = dataframe['text'].fillna('missing')
    if books:
        rows = dataframe[dataframe['orig_book_index'].str.contains('|'.join(books))]
        rows = rows[rows['text'].str.contains(phrase)]
    else:
        rows = dataframe[dataframe['text'].str.contains(phrase)]
    rows['search'] = rows['text_isopsephy'].apply(lambda numbers: find_cumulative_indices(numbers, isopsephy(phrase)))
    return rows

def get_verse(book, chapter, verse, dataframe):
    return ''.join(dataframe[dataframe['orig_book_index']==book][dataframe['orig_chapter']==chapter][dataframe['orig_verse']==verse].text)

def list_rows(rows):
    ul = h.ul()
    for idx, row in rows.iterrows():
        words = row.text.split(" ")
        for i in row.search:
            for j in i:
                words[j] = str(h.b(words[j], style="color: brown; text-decoration: underline; cursor: pointer", title=to_roman(words[j])))
        reference = h.span(h.b("%s %s:%s" % (row['orig_book_index'], row.orig_chapter, row.orig_verse)))
        ul += h.li(reference, " - ", h.span(' '.join(words)))

    return HTML(str(ul))

def levenshtein(source, target):
    # https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
    if len(source) < len(target):
        return levenshtein(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return previous_row[-1]

def strs_intersection(l):
    # intersection of multiple strings
    def intersection(s1, s2):
        matches = difflib.SequenceMatcher(None, s1, s2).get_matching_blocks()
        return ''.join([s1[x.a:x.a+x.size] for x in matches])
    # first string
    s = l[0]
    # latter strings
    for t in l[1:]:
        s = intersection(s, t)
    return s

def strs_comparison(l):
    # strs_comparison([str1, str2, str3]) -> (str, steps, percentage)
    if len(l) < 1:
        return None, None
    s = strs_intersection(l)
    steps = levenshtein(l[0], s)
    return (s, steps, float("{0:.2f}".format(100 - (steps/len(s))*100.)))

def get_variations(l = ['byzantine_2000', 'sblgnt', 
                        'tischendorf', 'modern',
                        'wh_ubs4', 'textus_receptus'],
                   filter_ = None):

    global manuscripts
    
    variations = {}

    for k in l:
        v = manuscripts[k]
        source = v['source']
        for var, title in v['variations'].items():
            k1 = "%s%s" % (k, "" if var == "var0" else "_%s" % var)
            data = load_dataframe(source, k, var)
            if filter_:
                for field, value in filter_.items():
                    data = data[data[field] == value]
            variations[k1] = {'data': data}
            variations[k1]['text'] = ''.join(variations[k1]['data'].text)
            variations[k1]['words_count'] = len(variations[k1]['text'].split(" "))
            variations[k1]['chars_count'] = len(variations[k1]['text'].replace(" ", ''))

    return variations

def diff_verses(text1, text2):
    return diff.diff_prettyHtml(diff.diff_main(text1, text2))

def diff_manuscripts(source = {'manuscript': 'textus_receptus', 'var': 'var1'},
                     target = {'manuscript': 'textus_receptus', 'var': 'var2'},
                     filter_ = {'orig_book_index': 'Revelation'}):
    diffs = []
    l = [source['manuscript'], target['manuscript']] if source['manuscript'] != target['manuscript'] else [source['manuscript']]
    variations = get_variations(l, filter_)
    var = source['var'] if source['var'] else "var0"
    source = "%s%s" % (source['manuscript'], "" if var == "var0" else "_%s" % var)
    var = target['var'] if target['var'] else "var0"
    target = "%s%s" % (target['manuscript'], "" if var == "var0" else "_%s" % var)
    m = max(variations[source]['data']['orig_chapter'])
    first_chapter = None
    for chapter in range(1, m+1):
        c1 = variations[source]['data'][variations[source]['data']['orig_chapter'] == chapter]
        c2 = variations[target]['data'][variations[target]['data']['orig_chapter'] == chapter]
        for verse in range(1, max(c1.orig_verse)+1):
            v1 = c1[c1.orig_verse == verse]
            v2 = c2[c2.orig_verse == verse]
            t1 = ''.join(v1.text).lower()
            t2 = ''.join(v2.text).lower()
            if t1 != t2:
                if first_chapter is None:
                    first_chapter = chapter
                    first_verse = verse
                diffs.append({'chapter': chapter, 
                              'verse': verse, 
                              'diff': diff_verses(t1, t2), 
                              'isopsephy': (sum(v1.total_isopsephy), sum(v2.total_isopsephy)), 
                              'chars': (len(t1.replace(" ", "")), len(t2.replace(" ", ""))), 
                              'words': (len(t1.split(" ")), len(t2.split(" ")))})
                last_chapter = chapter
                last_verse = verse
    d = "<br/><br/>".join(["<b>%s:%s</b> Words: %d | Chars: %d | Isopsephy: %d - %d = %d<br/><br/>%s" % \
                         (d['chapter'], 
                          d['verse'], 
                          d['words'][1]-d['words'][0], 
                          d['chars'][1]-d['chars'][0], 
                          d['isopsephy'][0],
                          d['isopsephy'][1],
                          d['isopsephy'][1]-d['isopsephy'][0],
                          d['diff']) for d in diffs])
    
    words = sum([d['words'][1]-d['words'][0] for d in diffs])
    chars = sum([d['chars'][1]-d['chars'][0] for d in diffs])
    isopsephy = sum([d['isopsephy'][1]-d['isopsephy'][0] for d in diffs])
    
    return """<h2>Differences between <b style="background-color: #e6ffe6">%s</b> %s:%s <> <b style="background-color: #ffe6e6">%s</b> %s:%s</h2>
              <h3>Instances: %d | Words: %d | Chars: %d | Isopsephy: %d</h3><br/><p>%s</p>""" % \
           (source, first_chapter, first_verse, target, last_chapter, last_verse, len(diffs), words, chars, isopsephy, d)
