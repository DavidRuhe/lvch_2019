"""Utilities for processing the results."""
import numpy as np
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm.auto import tqdm
from lime.lime_text import LimeTextExplainer


def plot_dendogram_and_tsne(hidden_dict, title, pca_components=128):
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

    data_link = linkage(squareform(distances), metric=None, method='ward')

    dendrogram(data_link, labels=np.array(list(hidden_dict.keys())), orientation='right')
    plt.suptitle(title, fontweight='bold', fontsize=14)
    plt.show()

    tsne = TSNE(n_components=2, verbose=1)
    tsne_results = tsne.fit_transform(np.concatenate(prob_matrices))

    plt.figure(figsize=(10, 10))
    i = 0
    for book in hidden_dict:
        plt.scatter(tsne_results[i:i + cases_per_book, 0],
                    tsne_results[i:i + cases_per_book, 1], label=book)
        i += cases_per_book

    plt.title("T-SNE plot of " + title)

    plt.legend()
    plt.show()


def explain_predictions(seeds, model, tokenizer,
                        num_lime_samples=1024,
                        num_explanations_per_book=8):
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

            if i == num_explanations_per_book:
                break
