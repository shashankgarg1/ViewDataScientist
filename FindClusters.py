import matplotlib.pyplot as plt
from datetime import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import DataLoader


def main():
    dataLoader = DataLoader.DataLoader("./Sensor_Weather_Data_Challenge.csv")
    df = dataLoader.getDf()
    # clusterDf = df.copy()
    clusterDf = df.iloc[:, 0: 14].copy()
    clusterDf["maxValue"] = clusterDf.iloc[:, 0:13].max(axis=1)
    clusterDf.drop(columns=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13"],
                   inplace=True)


    scaler = MinMaxScaler()
    x_scaled = scaler.fit_transform(clusterDf)
    clusterDf = pd.DataFrame(data=x_scaled, index=clusterDf.index)
    print(clusterDf.head())

    maxCluster = 11
    distortions = []
    for cluster in range(1, maxCluster):
        km = KMeans(
            n_clusters=cluster,
            init='random',
            n_init=30,
            max_iter=300,
            random_state=0)
        km.fit(clusterDf.values)
        distortions.append(km.inertia_)
    distortions = [None, None] + distortions[1:]
    plt.plot(range(1, maxCluster), distortions[1:], marker='o')
    plt.xlabel('No. of Clusters')
    plt.ylabel('Distortion')
    plt.title("K-means clusteing")
    plt.show()

    startDate = dt.strptime("2019-03-25".split(' ')[0], '%Y-%m-%d')
    endDate = startDate.replace(hour=23, minute=59, second=59, month=4)
    dateDf = clusterDf.loc[startDate: endDate]

    # plt.scatter(dateDf.iloc[:, 1], dateDf.iloc[:, 0])
    # plt.show()

    n_components = pd.np.arange(1, 10)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(clusterDf.values) for n in n_components]
    plt.plot(n_components, [m.bic(clusterDf.values) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(clusterDf.values) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('n_components');
    plt.show()

    gmm = GaussianMixture(n_components=4)
    gmm.fit(clusterDf.values)
    labels = gmm.predict(dateDf.values)
    plt.scatter(dateDf.values[:, 1], dateDf.values[:, 0], c=labels, cmap='viridis');
    plt.show()

    km = KMeans(
        n_clusters=4,
        init='random',
        n_init=30,
        max_iter=300,
        random_state=0)
    km.fit(clusterDf.values)
    labels = km.predict(dateDf.values)
    plt.scatter(dateDf.values[:, 1], dateDf.values[:, 0], c=labels, cmap='viridis');
    plt.show()




if __name__ == "__main__":
    main()
