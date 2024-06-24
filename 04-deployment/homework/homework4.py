
import pickle
import sys
import pandas as pd
from sklearn.metrics import mean_squared_error

with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)

categorical = ['PULocationID', 'DOLocationID']


def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def ride_duration_prediction(taxi_type, year, month):

    year = year
    month = month
    taxi_type = taxi_type

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'output/{taxi_type}/{year:04d}-{month:02d}.parquet'

    df = read_data(input_file)
    #df.head()
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    target = 'duration'
    y_train = df[target].values

    mean_squared_error(y_train, y_pred, squared=False)

    print("The standard deviation of predicted ride duration is ", y_pred.std())

    # Create unique ID for each record
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    # Extract the ride_id and y_pred into a new dataframe.
    df_result = pd.DataFrame({"ride_id": df["ride_id"], "y_pred" : y_pred})

    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    # Print the mean predicted duration
    print("The MEAN of predicted ride duration is ",df_result["y_pred"].mean())

def run():
    taxi_type = sys.argv[1] # 'green'
    year = int(sys.argv[2]) # 2021
    month = int(sys.argv[3]) # 3

    ride_duration_prediction(
        taxi_type=taxi_type,
        year=year, 
        month=month)

if __name__ == '__main__':
    run()
