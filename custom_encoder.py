from sklearn.base import BaseEstimator, TransformerMixin


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, data_sep=",", col_name_sep="_"):
        self.data_sep = data_sep
        self.col_name_sep = col_name_sep

    def fit(self, X, y=None):
        object_cols = X.select_dtypes(include="object").columns
        self.dummy_cols = [
            col
            for col in object_cols
            if X[col].str.contains(self.data_sep, regex=True).any()
        ]
        self.dummy_prefix = [
            "".join(map(lambda x: x[0], col.split(self.col_name_sep)))
            if self.col_name_sep in col
            else col[:2]
            for col in self.dummy_cols
        ]

        for col, pre in zip(self.dummy_cols, self.dummy_prefix):
            dummy_X = X.join(
                X[col]
                .str.get_dummies(sep=self.data_sep)
                .add_prefix(pre + self.col_name_sep)
            )

        dummy_X.drop(columns=self.dummy_cols, inplace=True)
        self.columns = dummy_X.columns
        return self

    def transform(self, X, y=None):
        for col, pre in zip(self.dummy_cols, self.dummy_prefix):
            X_transformed = X.join(
                X[col]
                .str.get_dummies(sep=self.data_sep)
                .add_prefix(pre + self.col_name_sep)
            )

        X_transformed = X_transformed.reindex(columns=self.columns, fill_value=0)
        return X_transformed

    # to get feature names
    def get_feature_names_out(self, input_features=None):
        return self.columns.tolist()
