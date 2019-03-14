"""When ran, extracts features from bible using BHSA."""
import os
from tf.app import use
import pandas as pd

from log import logger

MAIN_DIR = './'

def _append_to_main_dict(data, row_dict):

    for k in row_dict:
        data[k].append(row_dict[k])

    return data

def main():
    """Writes features to corpora folder."""
    use('bhsa', hoist=globals(), check=True)

    data = {
        'book_idx': [],
        'book': [],
        'chapter': [],
        'verse': [],
        'clause': [],
        'word': [],
        'lexeme': [],
        'word_pos': [],
        'verbal_stem': [],
        'word_number': [],
        'verbal_tense': [],
        'clause_type': [],
    }

    all_books = {T.bookName(b).lower(): i for i, b in enumerate(F.otype.s('book'))}

    for book_name in all_books:
        book_idx = all_books[book_name]

        logger.info(f"Extracting {book_name}...")

        b = F.otype.s('book')[book_idx]

        for i, c in enumerate(L.d(b, 'chapter')):

            for j, v in enumerate(L.d(c, 'verse')):

                for k, cl in enumerate(L.d(v, 'clause')):

                    for w in L.d(cl, 'word'):

                        row_dict = {
                            'book_idx': book_idx,
                            'book': book_name,
                            'chapter': i,
                            'verse': j,
                            'clause': k,
                            'word': T.text(w).strip(),
                            'lexeme': F.lex_utf8.v(w),
                            'word_pos': F.sp.v(w),
                            'verbal_stem': F.vs.v(w),
                            'word_number': F.nu.v(w),
                            'verbal_tense': F.vt.v(w),
                            'clause_type': F.typ.v(cl),
                            }

                        data = _append_to_main_dict(data, row_dict)

    data_df = pd.DataFrame(data)
    data_df.to_csv(os.path.join(MAIN_DIR, 'corpora', 'main_corpus.csv'), index=False)

if __name__ == '__main__':
    main()
