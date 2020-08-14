"""
DOCSTRING
"""
import catboost
import google.colab
import matplotlib.pyplot as pyplot
import numpy
import pandas
import sklearn

class EarthquakePrediction:
    """
    DOCSTRING
    """
    def __init__(self):
        uploaded = google.colab.files.upload()
        for fn in uploaded.keys():
            log = 'User uploaded file "{name}" with length {length} bytes'
            print(log.format(name=fn, length=len(uploaded[fn])))
        train = pandas.read_csv(
            'train.csv', nrows=6000000,
            dtype={'acoustic_data': numpy.int16, 'time_to_failure': numpy.float64})
        train_df.head(10)
        train_ad_sample_df = train['acoustic_data'].values[::100]
        train_ttf_sample_df = train['time_to_failure'].values[::100]

    def __call__(self):
        self.plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)
        train = pandas.read_csv(
            'train.csv', iterator=True, chunksize=150_000,
            dtype={'acoustic_data': numpy.int16, 'time_to_failure': numpy.float64})
        X_train = pandas.DataFrame()
        y_train = pandas.Series()
        for df in train:
            ch = self.gen_features(df['acoustic_data'])
            X_train = X_train.append(ch, ignore_index=True)
            y_train = y_train.append(pandas.Series(df['time_to_failure'].values[-1]))
        X_train.describe()
        # model #1 - CatBoost
        train_pool = catboost.Pool(X_train, y_train)
        regressor = catboost.CatBoostRegressor(
            iterations=10000, loss_function='MAE', boosting_type='Ordered')
        regressor.fit(X_train, y_train, silent=True)
        regressor.best_score_
        # model #2 - support vector machine w/ RBF + grid search
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(X_train)
        X_train_scaled = scaler.transform(X_train)
        parameters = [
            {'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],
             'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]
        reg1 = sklearn.model_selection.GridSearchCV(
            sklearn.svm.SVR(kernel='rbf', tol=0.01), parameters, cv=5,
            scoring='neg_mean_absolute_error')
        reg1.fit(X_train_scaled, y_train.values.flatten())
        y_pred1 = reg1.predict(X_train_scaled)
        print("Best CV score: {:.4f}".format(reg1.best_score_))
        print(reg1.best_params_)
        
    def gen_features(self, X):
        """
        Function to generate statistical features based on the training data.
        """
        strain = list()
        strain.append(X.mean())
        strain.append(X.std())
        strain.append(X.min())
        strain.append(X.max())
        strain.append(X.kurtosis())
        strain.append(X.skew())
        strain.append(numpy.quantile(X,0.01))
        strain.append(numpy.quantile(X,0.05))
        strain.append(numpy.quantile(X,0.95))
        strain.append(numpy.quantile(X,0.99))
        strain.append(numpy.abs(X).max())
        strain.append(numpy.abs(X).mean())
        strain.append(numpy.abs(X).std())
        return pandas.Series(strain)

    def plot_acc_ttf_data(
        self,
        train_ad_sample_df,
        train_ttf_sample_df,
        title="Acoustic data and time to failure: 1% sampled data"):
        """
        DOCSTRING
        """
        fig, ax1 = pyplot.subplots(figsize=(12, 8))
        pyplot.title(title)
        pyplot.plot(train_ad_sample_df, color='r')
        ax1.set_ylabel('acoustic data', color='r')
        pyplot.legend(['acoustic data'], loc=(0.01, 0.95))
        ax2 = ax1.twinx()
        pyplot.plot(train_ttf_sample_df, color='b')
        ax2.set_ylabel('time to failure', color='b')
        pyplot.legend(['time to failure'], loc=(0.01, 0.9))
        pyplot.grid(True)

if __name__ == '__main__':
    earthquake_prediction = EarthquakePrediction()
    earthquake_prediction()
