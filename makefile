.DEFAULT_GOAL := artifacts/model.pkl

artifacts/model.pkl: train_draft_position_predictor.py artifacts/preprocessed.csv
	C:/Users/mrquo/anaconda3/envs/ml/python.exe \
	c:/Users/mrquo/Desktop/School/2023SpringSummer/SIADS696/nhl-draft-predictions/train_draft_position_predictor.py -v \
	artifacts/preprocessed.csv \
	artifacts/model.pkl \
	artifacts/metrics.txt
	

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

