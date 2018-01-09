from whoosh.analysis import StemFilter, StandardAnalyzer, StemmingAnalyzer, Filter, LowercaseFilter, StopFilter, RegexTokenizer, Tokenizer, Token
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.qparser import QueryParser, MultifieldParser
from whoosh import qparser, query, sorting
from nltk.tokenize import sent_tokenize
from nltk.stem import WordNetLemmatizer
from whoosh.scoring import BM25F
from nltk import pos_tag
from nltk.corpus import wordnet as wn
import whoosh.index as index
import os
import itertools
import whoosh
import sys
import spacy


def main():
    file_content_doc1 = open("rural_min.txt").read()
    file_content_doc2 = open("science_min.txt").read()
    option = True;
    while option:
        print("""
        1. Create Index.
        2. Query Index.
        3. Exit
        """)
        option=input("Please select an option...!")
        if option=="1":

            sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
            sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')
            if not os.path.exists("index_task3_min"):
                os.mkdir("index_task3_min")

            my_analyzer = RegexTokenizer() | StopFilter() | LowercaseFilter() | Lemmatizer()
            pos_tagger = RegexTokenizer() | StopFilter() | LowercaseFilter() | PosTagger()
            wordnetsyn1 = RegexTokenizer() | StopFilter() | LowercaseFilter() | WordNetSynsets()
            wordnetsyn2 = RegexTokenizer() | StopFilter() | LowercaseFilter() | WordNetSynsets1()
            wordnetsyn3 = RegexTokenizer() | StopFilter() | LowercaseFilter() | WordNetSynsets2()
            wordnetsyn4 = RegexTokenizer() | StopFilter() | LowercaseFilter() | WordNetSynsets3()


            schema = Schema(id=ID(stored=True, unique=True), standard=TEXT(stored=True, analyzer=StandardAnalyzer()),
                            stem_text = TEXT(stored=True, analyzer=StemmingAnalyzer()), lemma = TEXT(stored=True, analyzer=my_analyzer),
                            pos_text = TEXT(stored=True, analyzer=pos_tagger), hypernym = TEXT(stored=True, analyzer=wordnetsyn1),
                            hyponym = TEXT(stored=True, analyzer=wordnetsyn2), holonym = TEXT(stored=True, analyzer=wordnetsyn3),
                            meronyms = TEXT(stored=True, analyzer=wordnetsyn4),
                            dependency = TEXT(analyzer=DependencyParser()))

            ix = index.create_in("index_task3_min", schema)
            writer = ix.writer()

            for sentence in sent_tokenize_list1:
                 writer.add_document(standard=sentence, stem_text=sentence, lemma=sentence, pos_text=sentence, hypernym=sentence,
                                hyponym=sentence, meronyms=sentence, holonym=sentence, dependency = sentence)
            for sentence in sent_tokenize_list2:
                writer.add_document(standard=sentence, stem_text=sentence, lemma=sentence, pos_text=sentence,
                                    hypernym=sentence,
                                    hyponym=sentence, meronyms=sentence, holonym=sentence, dependency=sentence)
            writer.commit()

            print_index_details(ix)

            print("\n\n Index created with various features as its fields")

        elif option=="2":
            ix = index.open_dir("index_task3")

            with ix.searcher(weighting=whoosh.scoring.BM25F()) as searcher:
                og = qparser.OrGroup.factory(0.5)
                q = input("\n Insert a query...!")
                query_text = MultifieldParser(["standard", "stem_text", "lemma", "pos_text","hyponym","meronyms", "hypernym","holonym"], schema=ix.schema, group=og).parse(q)
                results = searcher.search(query_text, limit=10)
                for i, hit in enumerate(results):
                    print(results.score(i), hit["standard"], sep=":")
                    print("\n")

        elif option == "3":
            print("\n Goodbye")
            sys.exit(0)
            option = None
        else:
            print("\n Not valid choice try again...!")

def print_index_details(ix):
    docs = ix.searcher().reader()
    stem, lemma = itertools.tee(docs)
    standard, hypernyms = itertools.tee(docs)
    holonyms, meronyms = itertools.tee(docs)
    headwords, postext = itertools.tee(docs)

    print('Stemming index')
    for doc in stem:
        if doc[0][0] == 'stem_text':
            print(doc)
    print("----------------------------------------------------------------------------------------------------------------------")

    print('\nLemmatizing index')
    for doc in lemma:
        if doc[0][0] == 'lemma':
            print(doc)
    print(
        "----------------------------------------------------------------------------------------------------------------------")

    print('\nPos tagging Index')
    for doc in postext:
        if doc[0][0] == 'pos_text':
            print(doc)
    print(
        "----------------------------------------------------------------------------------------------------------------------")

    print('\nDependency parse relations')
    for doc in headwords:
        if doc[0][0] == 'dependency':
            print(doc)

    print("----------------------------------------------------------------------------------------------------------------------")
    print('\nHypernym index')
    for doc in hypernyms:
        if doc[0][0] == 'hypernym':
            print(doc)

    print("----------------------------------------------------------------------------------------------------------------------")

    print('\nHyponym index')
    for doc in docs:
        if doc[0][0] == 'hyponym':
            print(doc)

    print(
        "----------------------------------------------------------------------------------------------------------------------")
    print('\nHolonym index')
    for doc in holonyms:
        if doc[0][0] == 'holonym':
            print(doc)

    print(
        "----------------------------------------------------------------------------------------------------------------------")
    print('\nMeronym index')
    for doc in meronyms:
        if doc[0][0] == 'meronyms':
            print(doc)


# filter for lemmatizing the data
class Lemmatizer(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        lm = WordNetLemmatizer()
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            if text in self.cache:
                t.text = self.cache[text]
                yield t
            else:
                lemma = lm.lemmatize(text)
                self.cache[t.text] = lemma
                t.text = lemma
                yield t


# filter for Pos Tagging
class PosTagger(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        words = []
        tokens1, tokens2 = itertools.tee(tokens)
        for t in tokens1:
            words.append(t.text)
        tags = pos_tag(words)
        i = 0
        for t in tokens2:
            t.text = tags[i][0] + " " + tags[i][1]
            i += 1
            yield t


class WordNetSynsets(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            for ss in wn.synsets(text):
                t.text = ', '.join([str(x) for x in ss.hypernyms()])
                yield t


class WordNetSynsets1(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            for ss in wn.synsets(text):
                t.text = ', '.join([str(x) for x in ss.hyponyms()])
                yield t


class WordNetSynsets2(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            for ss in wn.synsets(text):
                t.text = ', '.join([str(x) for x in ss.member_holonyms()])
                yield t


class WordNetSynsets3(Filter):
    def __eq__(self, other):
        return (other
                and self.__class__ is other.__class__
                and self.__dict__ == other.__dict__)

    def __ne__(self, other):
        return not self == other

    def __init__(self):
        self.cache = {}

    def __call__(self, tokens):
        assert hasattr(tokens, "__iter__")
        for t in tokens:
            if t.stopped:
                yield t
                continue
            text = t.text
            for ss in wn.synsets(text):
                t.text = ', '.join([str(x) for x in ss.part_meronyms()])
                yield t


class DependencyParser(Tokenizer):
    def __eq__(self, other):
        return other and self.__class__ is other.__class__

    def __call__(self, value, positions=False, chars=False, keeporiginal=False,
                     removestops=True, start_pos=0, start_char=0, tokenize=True,
                     mode='', **kwargs):
        t = Token(positions, chars, removestops=removestops, mode=mode,
                      **kwargs)
        nlp = spacy.load('en_core_web_sm')
        doc = nlp(value)

        t.pos = start_pos

        for chunk in doc:
            t.text = chunk.dep_
            yield t

if __name__ == "__main__":
    main()