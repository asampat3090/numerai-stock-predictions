# Numerai Stock Predictions

[Numer.ai](https://numer.ai/) is a hedge fund that hosts weekly online competitions for data science model building, allowing data science enthuasists a way to earn money by building models that help to explain phenomena in their sanitized data. A great part about working with Numer.ai data is that it’s already curated for you — no need to deal with wrangling, scraping, or cleaning. This is pretty awesome, as data cleaning is one of the more frustrating/boring parts of machine learning.

This repository uses the Numerai data from July 22nd and contains a base model built using the XGBoost algorithm.

# Dockerfile
Included is a Dockerfile that you can use to start up a container with Docker 

## Analysis
To run through the analysis and experiment you can startup up the Jupyter notebook

```bash
$ jupyter notebook
```

## Predictions
 If you want to run the python script by itself rather than a Jupyter notebook, run the following
 
```bash
$ python example_model.py
```


