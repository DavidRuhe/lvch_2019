"""When ran, extracts features from bible using BHSA."""
import os
from tf.app import use
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

MAIN_DIR = './'


def main():
    """Writes features to corpora folder."""
    use('bhsa', hoist=globals(), check=True)

    all_books = {T.bookName(b).lower(): i for i, b in enumerate(F.otype.s('book'))}

    for i, book_name in enumerate(all_books):
        book_idx = all_books[book_name]

        logger.info(f"Extracting {book_name}...")

        b = F.otype.s('book')[book_idx]
        book = []
        for v in L.d(b, 'verse'):

            verse = []
            for s in L.d(v, 'sentence'):

                sentence = []
                for w in L.d(s, 'word'):

                    word = T.text(w).strip()

                    if len(word) == 0:
                        continue

                    sentence.append(word)

                if len(sentence) > 0:

                    sentence.insert(0, 'eos')
                    verse.append(sentence)

            if len(verse) > 0:
                verse.insert(0, ['eov'])
                book.append(verse)

        book = ' '.join(word for verse in book for sentence in verse for word in sentence)

        with open(os.path.join(MAIN_DIR, 'corpora', 'word', '%s.txt' % book_name), 'wb') as f:
            f.write(book.encode())


if __name__ == '__main__':
    main()
