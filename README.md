# COSC4337-Project
Group Project for COSC 4337 Data Science II

## Dataset

This project uses the [US Accidents dataset](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents) from Kaggle, which covers traffic accident records across the United States.

## Getting Started

### 1. Download the Dataset

1. Go to the [US Accidents Kaggle page](https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)
2. Sign in to your Kaggle account (or create one for free)
3. Click **Download** to download `US_Accidents_March23.csv` (or the current version)

### 2. Filter for Houston, Dallas, and Austin

Once the dataset is downloaded, filter it to only include accidents from the three target Texas cities. You can do this with pandas:
```python
import pandas as pd

df = pd.read_csv('US_Accidents_March23.csv')

cities = ['Houston', 'Dallas', 'Austin']

for city in cities:
    city_df = df[df['City'] == city]
    city_df.to_csv(f'{city.lower()}_data.csv', index=False)
```

This will produce 3 filtered CSV files containing only records from Houston, Dallas, and Austin, which are used as the input for the rest of the project.
