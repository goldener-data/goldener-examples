import numpy as np
from sklearn.base import _fit_context
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer


class NormmalizedKMeans(KMeans):
    def __init__(
        self,
        n_clusters=8,
        *,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=1e-4,
        verbose=0,
        random_state=None,
        copy_x=True,
        algorithm="lloyd",
    ):
        super().__init__(
            n_clusters=n_clusters,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
        )
        self.normalizer = Normalizer()

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        sample_weight: None | np.ndarray = None,
    ) -> np.ndarray:
        return super().fit(self.normalizer.transform(X), y, sample_weight)

    def fit_predict(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        sample_weight: None | np.ndarray = None,
    ) -> np.ndarray:
        return super().fit_predict(self.normalizer.transform(X), y, sample_weight)

    def predict(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        sample_weight: None | np.ndarray = None,
    ) -> np.ndarray:
        return super().predict(self.normalizer.transform(X), y, sample_weight)

    def fit_transform(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        sample_weight: None | np.ndarray = None,
    ) -> np.ndarray:
        return super().fit_transform(self.normalizer.transform(X), y, sample_weight)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return super().transform(self.normalizer.transform(X))

    def score(
        self,
        X: np.ndarray,
        y: None | np.ndarray = None,
        sample_weight: None | np.ndarray = None,
    ) -> np.ndarray:
        return super().score(self.normalizer.transform(X), y, sample_weight)
