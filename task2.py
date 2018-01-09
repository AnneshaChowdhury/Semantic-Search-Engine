
from nltk.tokenize import sent_tokenize
from whoosh.index import create_in
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StandardAnalyzer
from whoosh.qparser import QueryParser
from whoosh import qparser
import whoosh.index as index
import os

def main():
    option = True
    while option:
        print("""
            1. Create Index.
            2. Query Index.
            3. Exit
            """)
        option = input("Please select an option...!")
        if option == "1":
            file_content_doc1 = open("rural.txt").read()
            file_content_doc2 = open("science.txt").read()
            sent_tokenize_list1 = sent_tokenize(file_content_doc1, language='english')
            sent_tokenize_list2 = sent_tokenize(file_content_doc2, language='english')
            if not os.path.exists("index_task2"):
                os.mkdir("index_task2")

            schema = Schema(full_text=TEXT(phrase=True, stored=True, analyzer=StandardAnalyzer(stoplist=None)))
            ix = create_in("index_task2", schema)
            writer = ix.writer()

            for sentence in sent_tokenize_list1:
                writer.add_document(full_text=sentence)
            for sentence in sent_tokenize_list2:
                writer.add_document(full_text=sentence)
            writer.commit()
            print("\n\n Index created with various features as its fields")
        elif option == "2":
            ix = index.open_dir("index_task2")
            with ix.searcher() as searcher:

                og = qparser.OrGroup.factory(0.5)
                q = input("\n Insert a query...!")
                query = QueryParser("full_text", ix.schema, group=og).parse(q)
                results = searcher.search(query)

                for i, hit in enumerate(results):
                    print(results.score(i), hit["full_text"], sep=":")
                    print("\n")


if __name__ == "__main__":
    main()