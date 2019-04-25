"""Utilities for processing the results."""
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from lime.lime_text import LimeTextExplainer
import seaborn as sns
from matplotlib import cm



def plot_dendogram_and_tsne(hidden_dict, title, predictions=None, pca_components=128):
    """Plot dendogram and t-sne plots of hidden dimensions.

    Parameters
    ----------
    hidden_dict: dictionary with book keys and hidden dims as values
    title: Title of plot.
    pca_components: number of pca components to reduce to.
    """

    cases_per_book = list(hidden_dict.values())[0].shape[0]

    pca = PCA(n_components=pca_components)
    pca.fit(np.concatenate([hidden_dict[book] for book in hidden_dict]))

    print('Variance PCA: {}'.format(np.sum(pca.explained_variance_ratio_)))

    assert np.sum(pca.explained_variance_ratio_) > 0.85

    for book in hidden_dict:
        hidden_dict[book] = pca.transform(hidden_dict[book])

    distances = np.empty((len(hidden_dict), len(hidden_dict)), dtype=np.float32)

    prob_matrices = [hidden_dict[book] for book in hidden_dict]

    for i in range(len(distances)):
        for j in range(i, len(distances)):

            mat_a = prob_matrices[i]
            mat_b = prob_matrices[j]

            distance = np.linalg.norm(mat_a - mat_b)

            distances[(i, j)] = distance
            distances[(i, j)[::-1]] = distance

    data_link = linkage(squareform(distances), metric=None, method='ward', optimal_ordering=True)

    dendrogram(data_link, labels=np.array(list(hidden_dict.keys())), orientation='right')
    plt.suptitle(title, fontweight='bold', fontsize=14)
    plt.show()

    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(np.concatenate(prob_matrices))

    plt.figure(figsize=(10, 10))
    i = 0
    cmap = sns.color_palette('rainbow', n_colors=len(hidden_dict))

    for j, book in enumerate(hidden_dict):

        sns.scatterplot(tsne_results[i:i + cases_per_book, 0],
                    tsne_results[i:i + cases_per_book, 1], label=book, color=cmap[j])
        i += cases_per_book

    plt.legend()
    plt.show()

    sbh = ['genesis', 'exodus', 'leviticus', 'deuteronomy', 'joshua',
           'judges', 'kings', 'samuel']
    lbh = ['song_of_songs', 'ecclesiastes', 'esther', 'daniel', 'ezra-nehemiah', 'chronicles']

    plt.figure(figsize=(10, 10))
    i = 0

    sbh_tsne = []
    lbh_tsne = []

    for j, book in enumerate(hidden_dict):
        if book in sbh:
            sbh_tsne.append(([tsne_results[i:i + cases_per_book]]))
        elif book in lbh:
            lbh_tsne.append(([tsne_results[i:i + cases_per_book]]))

        else:
            raise KeyError(f"{book} not in sbh or lbh.")

        i += cases_per_book

    sbh_tsne = np.hstack(sbh_tsne).squeeze(0)
    lbh_tsne = np.hstack(lbh_tsne).squeeze(0)

    plt.scatter(sbh_tsne[:, 0], sbh_tsne[:, 1], label='sbh', color='blue')
    plt.scatter(lbh_tsne[:, 0], lbh_tsne[:, 1], label='lbh', color='red')

    plt.legend()
    plt.show()

    if predictions:
        all_predictions = np.concatenate([predictions[b] for b in predictions])
        unique_predicitons = np.unique(all_predictions)

        pred_cmap = sns.color_palette('rainbow', n_colors=len(unique_predicitons))

        plt.figure(figsize=(10, 10))

        for i, prediction in enumerate(unique_predicitons):
            pred_tsnes = tsne_results[all_predictions == prediction]
            plt.scatter(pred_tsnes[:, 0], pred_tsnes[:, 1], label=prediction, color=pred_cmap[i])

    plt.legend()
    plt.show()

    prose = ['genesis', 'exodus', 'leviticus', 'deuteronomy', 'joshua', 'judges', 'samuel', 'kings',
             'daniel', 'ezra-nehemiah', 'chronicles', 'esther']

    poetry = ['song_of_songs', 'ecclesiastes']

    plt.figure(figsize=(10, 10))
    i = 0

    prose_tsne = []
    poetry_tsne = []

    for j, book in enumerate(hidden_dict):
        print(book)
        if book in prose:
            prose_tsne.append(([tsne_results[i:i + cases_per_book]]))
        elif book in poetry:
            poetry_tsne.append(([tsne_results[i:i + cases_per_book]]))

        else:
            raise KeyError(f"{book} not in prose or poetry.")

        i += cases_per_book

    prose_tsne = np.hstack(prose_tsne).squeeze(0)

    if len(poetry_tsne) > 0:
        poetry_tsne = np.hstack(poetry_tsne).squeeze(0)
        plt.scatter(poetry_tsne[:, 0], poetry_tsne[:, 1], label='poetry', color='red')

    plt.scatter(prose_tsne[:, 0], prose_tsne[:, 1], label='prose', color='blue')

    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 10))

    i = 0
    alpha = 0.3

    for j, book in enumerate(hidden_dict):

        sns.kdeplot(tsne_results[i:i + cases_per_book, 0],
                    tsne_results[i:i + cases_per_book, 1], label=book, color=cmap[j],
                    n_levels=1, alpha=alpha, shade=True, shade_lowest=False)

        i += cases_per_book


    leg = plt.legend()
    for lh in leg.legendHandles:
        lh.set_alpha(alpha)

    plt.show()


    plt.figure(figsize=(10, 10))
    i = 0
    for j, book in enumerate(hidden_dict):

        sns.kdeplot(tsne_results[i:i + cases_per_book, 0],
                    tsne_results[i:i + cases_per_book, 1], label=book, color=cmap[j],
                    n_levels=1)

        i += cases_per_book


    plt.legend()

    plt.show()


def explain_predictions(seeds, model, tokenizer,
                        num_lime_samples=1024,
                        num_explanations_per_book=2):
    """Explains keras language model given a dictionary of encoded seeds.
    Parameters
    ----------
    seeds: dict
        Dictionary of keys (books) and seeds.
    model: keras.Model
        Keras classifier
    tokenizer: Tokenizer
        Tokenizer instance (see utils)
    num_lime_samples: int
        Number of samples to use in the lime algorithm.
    num_explanations_per_book: int
        How many explanations per book

    Returns
    -------
    Shows explanations in Jupyter notebook.
    """

    def _predict_fn(input_text):

        input_encoded = tokenizer.encode(input_text, errors='ignore')
        predictions = np.empty([len(input_encoded), tokenizer.num_words])

        for j, x_in in enumerate(tqdm(input_encoded)):
            model.reset_states()
            for w in x_in:
                x = np.array(w)[np.newaxis]

                y_ = model.predict(x)

            predictions[j] = y_

        return predictions

    for book in seeds:
        print("\n")
        print("------------------------------\n")
        print(f"Explanations for {book}")
        explainer = LimeTextExplainer(class_names=list(tokenizer.word_to_ix.keys()))

        for i, sample in enumerate(seeds[book]):

            encoded_data = sample[:-1]
            encoded_label = sample[-1]
            text = tokenizer.decode(encoded_data)

            explanation = explainer.explain_instance(text,
                                                     _predict_fn,
                                                     labels=(encoded_label,),
                                                     num_samples=num_lime_samples)

            explanation.show_in_notebook()

            if i == num_explanations_per_book - 1:
                break
