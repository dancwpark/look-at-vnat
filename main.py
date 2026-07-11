import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def main():
    # exploring the dataset
    df = pd.read_hdf("data/VNAT_Feature_Dataframe_release_1.h5")

    ## columns
    for i in df.columns.values.tolist():
        print(i)
    print(len(df.columns.values.tolist()))
    ## the last column is the labels -- 129 features per sample
    df_features = df.iloc[:, :129]
    df_labels = df.iloc[:, 129:]

    print(df_features.columns.values.tolist())
    print(df_labels.columns.values.tolist())

    np.random.seed(42)
    rndperm = np.random.permutation(df_features.shape[0])

    # quick viz using pca - 2 vecs
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(df_features.values)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
            x=pca_result[rndperm, 0],
            y=pca_result[rndperm, 1],
            hue=df_labels.iloc[rndperm, 0],
            palette="hls",
            legend="full",
            alpha=0.3)
    plt.show() 

    # quick viz using pca - 3 vecs
    labels = df_labels.iloc[rndperm, 0]
    label_codes, uniques = pd.factorize(labels)
    assert len(uniques) > 1, "Need more than one class to plot"
    colors = plt.cm.hsv(label_codes / label_codes.max())
    ax = plt.figure(figsize=(16, 10)).add_subplot(111, projection='3d')
    ax.scatter(
            xs=pca_result[rndperm, 0],
            ys=pca_result[rndperm, 1],
            zs=pca_result[rndperm, 2],
            c=colors,
            alpha=0.3,
            )
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=plt.cm.hsv(i / len(uniques)),
                          markersize=8, label=uniques[i])
               for i in range(len(uniques))]
    ax.legend(handles=handles, title='Label', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    # tSNE
    #tsne = TSNE(n_components=3, verbose=1, perplexity=30, max_iter=1000)
    tsne = TSNE(n_components=3, verbose=1, perplexity=50, max_iter=1500)
    #tsne = TSNE(n_components=3, verbose=1, perplexity=60, max_iter=2000)
    tsne_results = tsne.fit_transform(df_features.values)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
            x=tsne_results[:, 0],
            y=tsne_results[:, 1],
            hue=df_labels.iloc[:, 0],
            palette='hls',
            legend='full',
            alpha=0.3)
    plt.show()

    # PCA (50 components) -> tSNE
    pca50 = PCA(n_components=50)
    pca50_result = pca50.fit_transform(df_features.values)
    tsne50 = TSNE(n_components=2, verbose=1, perplexity=50, max_iter=1500)
    tsne50_results = tsne50.fit_transform(pca50_result)
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
            x=tsne50_results[:, 0],
            y=tsne50_results[:, 1],
            hue=df_labels.iloc[:, 0],
            palette='hls',
            legend='full',
            alpha=0.3)
    plt.show()



if __name__ == "__main__":
    main()
