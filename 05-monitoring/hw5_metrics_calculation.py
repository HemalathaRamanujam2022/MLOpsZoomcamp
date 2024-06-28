import datetime
import time
import random
import logging 
import uuid
import pytz
import pandas as pd
import io
import psycopg
import joblib


from evidently.report import Report
from evidently import ColumnMapping
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnQuantileMetric, RegressionQualityMetric
from evidently.metrics import DatasetSummaryMetric,  ConflictTargetMetric
from evidently.metrics import RegressionPredictedVsActualPlot, RegressionTopErrorMetric
from evidently.metrics import RegressionPredictedVsActualScatter, ColumnCorrelationsMetric

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

SEND_TIMEOUT = 10
rand = random.Random()

create_table_statement = """
drop table if exists dummy_metrics_hw;
create table dummy_metrics_hw(
	timestamp timestamp,
	prediction_drift float,
	num_drifted_columns integer,
	share_missing_values float,
	fare_amount_quantile float
)
"""

reference_data = pd.read_parquet('data/reference.parquet')
with open('models/lin_reg.bin', 'rb') as f_in:
	model = joblib.load(f_in)

raw_data = pd.read_parquet('data/green_tripdata_2024-03.parquet')

raw_data["duration_min"] = raw_data.lpep_dropoff_datetime - raw_data.lpep_pickup_datetime
raw_data.duration_min = raw_data.duration_min.apply(lambda td : float(td.total_seconds())/60)

print("Shape of raw data for March, 2024 : ", raw_data.shape)

begin = datetime.datetime(2024, 3, 1, 0, 0)
num_features = ['passenger_count', 'trip_distance', 'fare_amount', 'total_amount']
cat_features = ['PULocationID', 'DOLocationID']
column_mapping = ColumnMapping(
    prediction='prediction',
    numerical_features=num_features,
    categorical_features=cat_features,
    # target=None
	target='duration_min'
)

report = Report(metrics = [
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric(),
	ColumnQuantileMetric(column_name="fare_amount", quantile=0.5),
	RegressionQualityMetric(),
	ColumnCorrelationsMetric(column_name="prediction")
# 	,DatasetSummaryMetric()
# 	,DatasetCorrelationsMetric()
# 	,ConflictTargetMetric(),
# 	,RegressionPredictedVsActualPlot()
])

def prep_db():
	with psycopg.connect("host=localhost port=5433 user=postgres password=example", autocommit=True) as conn:
		res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
		if len(res.fetchall()) == 0:
			conn.execute("create database test;")
		with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example") as conn:
			conn.execute(create_table_statement)

def calculate_dummy_metrics_postgresql(curr):
	value1 = rand.randint(0, 1000)
	value2 = str(uuid.uuid4())
	value3 = rand.random()

	curr.execute(
		"insert into dummy_metrics_hw(timestamp, value1, value2, value3) values (%s, %s, %s, %s)",
		(datetime.datetime.now(pytz.timezone('Europe/London')), value1, value2, value3)
	)

def calculate_metrics_postgresql(curr, i):
	current_data = raw_data[(raw_data.lpep_pickup_datetime >= (begin + datetime.timedelta(i))) &
		(raw_data.lpep_pickup_datetime < (begin + datetime.timedelta(i + 1)))]

	#current_data.fillna(0, inplace=True)
	current_data['prediction'] = model.predict(current_data[num_features + cat_features].fillna(0))

	report.run(reference_data = reference_data, current_data = current_data,
		column_mapping=column_mapping)

	result = report.as_dict()
	# print("Report result " , result)

	prediction_drift = result['metrics'][0]['result']['drift_score']
	num_drifted_columns = result['metrics'][1]['result']['number_of_drifted_columns']
	share_missing_values = result['metrics'][2]['result']['current']['share_of_missing_values']
	fare_amt_qntl = result['metrics'][3]['result']['current']['value']
	# print("fare amount quantile (Daily) :",	 fare_amt_qntl)
	# print("RegressionQualityMetric : current : rmse", result['metrics'][4]['result']['current']['rmse'])
	# print("RegressionQualityMetric : reference : rmse", result['metrics'][4]['result']['reference']['rmse'])
	# print("ColumnCorrelationsMetric ", result['metrics'][5]['result']['current']['pearson'])

	# print("DatasetSummaryMetric  : ",result['metrics'][4]['result'] )
	# print("RegressionPredictedVsActualPlot  : ",result['metrics'][5]['result'] )

	curr.execute(
		"insert into dummy_metrics_hw(timestamp, prediction_drift, num_drifted_columns, share_missing_values, fare_amount_quantile) values (%s, %s, %s, %s, %s)",
		(begin + datetime.timedelta(i), prediction_drift, num_drifted_columns, share_missing_values, fare_amt_qntl)
	)

def main():
	prep_db()
	last_send = datetime.datetime.now() - datetime.timedelta(seconds=10)
	with psycopg.connect("host=localhost port=5433 dbname=test user=postgres password=example", autocommit=True) as conn:
		# for i in range(0, 100):
		# 	with conn.cursor() as curr:
		# 		calculate_dummy_metrics_postgresql(curr)

		# 	new_send = datetime.datetime.now()
		# 	seconds_elapsed = (new_send - last_send).total_seconds()
		# 	if seconds_elapsed < SEND_TIMEOUT:
		# 		time.sleep(SEND_TIMEOUT - seconds_elapsed)
		# 	while last_send < new_send:
		# 		last_send = last_send + datetime.timedelta(seconds=10)
		# 	logging.info("data sent")

		# for i in range(0, 31):
		for i in range(0,31):
			with conn.cursor() as curr:
				calculate_metrics_postgresql(curr, i)

			new_send = datetime.datetime.now()
			seconds_elapsed = (new_send - last_send).total_seconds()
			if seconds_elapsed < SEND_TIMEOUT:
				time.sleep(SEND_TIMEOUT - seconds_elapsed)
			while last_send < new_send:
				last_send = last_send + datetime.timedelta(seconds=10)
			logging.info("data sent")

if __name__ == '__main__':
	main()