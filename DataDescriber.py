import DataLoader
import pandas as pd

def main():
    dataLoader = DataLoader.DataLoader("./Sensor_Weather_Data_Challenge.csv")
    df = dataLoader.getDf()
    newDf = pd.DataFrame(columns=df["d1"].describe().index)
    for column in df.columns:
        print (column)
    for column in df.columns.values:
        print("The statistics for column: ", column, " is below:")
        print(df[column].describe())
        newDf = newDf.append(df[column].describe())
        print(newDf)

    print(len(df.columns))


if __name__ == "__main__":
    main()
