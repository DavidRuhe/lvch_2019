"""Extracts the King James bible into the standard format."""
import numpy as np
import pandas as pd

from constants import KJ_MAPPER


def explode(df, lst_cols, fill_value='', preserve_index=False):
    """Explodes the specified column values (lists) to new rows.

    Parameters
    ----------
    df: pd.DataFrame
    lst_cols: list of columns to explode
    fill_value: str
    preserve_index: bool

    Returns
    -------
    pd.DataFrame
    """

    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
            and len(lst_cols) > 0 and not isinstance(lst_cols,
                                                     (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
        col: np.repeat(df[col].values, lens)
        for col in idx_cols},
        index=idx)
           .assign(**{col: np.concatenate(df.loc[lens > 0, col].values)
                      for col in lst_cols if len(df.loc[lens > 0, col].values) > 0}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens == 0, idx_cols], sort=False)
               .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:
        res = res.reset_index(drop=True)
    return res


def main():
    """Writes the king james bible to the csv format."""

    bible = pd.read_csv('./corpora/kjvdat.txt', sep='|', header=None, names=['book',
                                                                             'verse',
                                                                             'phrase',
                                                                             'text'])

    bible['text'] = bible['text'].str.replace('~', ' eos').str.replace("[^\w\s]", '')

    bible['book'] = bible['book'].map(KJ_MAPPER)

    bible = bible.dropna(subset=['book'])

    bible['text'] = bible['text'].str.split(' ')
    bible = explode(bible, ['text'])

    bible['word'] = bible['text']
    bible = bible.drop('text', axis=1)

    bible['word'] = bible['word'].replace('', np.nan)

    bible = bible.dropna(subset=['word'])

    bible.to_csv('corpora/english_corpus.csv')


if __name__ == '__main__':
    main()
