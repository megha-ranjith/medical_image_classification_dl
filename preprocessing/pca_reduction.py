from sklearn.decomposition import PCA

def apply_pca(features, n_components=128):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(features)
    return reduced, pca
