import pandas as pd

df = pd.read_csv('./assign01/data/amazon.csv')

filtered_df = df[(df['stars'] > 0) & (df['reviews'] > 0)]

count = filtered_df.shape[0]

filtered_df.to_csv('./assign01/data/amazon_cleaned.csv')
print("Number of records with stars and reviews greater than 0:", count)
