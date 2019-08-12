import pandas as pd


class DataLoader():
    def __init__(self, location):
        self._df = pd.read_csv(location)
        self._df.set_index(pd.DatetimeIndex(data=self._df["Time"]), inplace=True)
        self._df.drop(columns=["Time"], inplace=True)

    def getDf(self):
        return self._df


def main():
    dataLoader = DataLoader("./Sensor_Weather_Data_Challenge.csv")
    df = dataLoader.getDf()
    print(df.columns.values)


if __name__ == "__main__":
    main()
