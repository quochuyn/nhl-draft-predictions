.DEFAULT_GOAL := artifacts/preprocessed.csv

artifacts/preprocessed.csv: preprocess_reports.py artifacts/clean.csv
	C:/Users/mrquo/anaconda3/envs/ml/python.exe \
	c:/Users/mrquo/Desktop/School/2023SpringSummer/SIADS696/nhl-draft-predictions/preprocess_reports.py \
	artifacts/clean.csv \
	artifacts/preprocessed.csv

artifacts/clean.csv: clean_reports.py data/prospect-data.csv
	C:/Users/mrquo/anaconda3/envs/ml/python.exe \
	c:/Users/mrquo/Desktop/School/2023SpringSummer/SIADS696/nhl-draft-predictions/clean_reports.py \
	data/prospect-data.csv \
	artifacts/clean.csv

