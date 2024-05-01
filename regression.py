import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm

df = pd.read_csv('./data/amazon_cleaned.csv')


star_weight = 0.8
review_weight = 0.2

df['Rating'] = (df['stars'] * star_weight) + (df['reviews'] * review_weight)

print(df[['stars', 'reviews', 'Rating']])


def plot_author_rating_vs_books_written():
    author_stats = df.groupby('author').agg(
        {'Rating': 'mean', 'asin': 'count'}).reset_index()
    author_stats = author_stats.rename(columns={'asin': 'books_written'})

    author_stats = author_stats.sort_values(by='books_written')

    plt.figure(figsize=(10, 6))
    plt.scatter(author_stats['books_written'],
                author_stats['Rating'], alpha=0.5, label='Data Points')

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(
        func, author_stats['books_written'], author_stats['Rating'])

    x_values = np.linspace(0, author_stats['books_written'].max(), 100)
    y_values = func(x_values, *popt)
    plt.plot(x_values, y_values, color='red', label='Curve Fit')

    X = sm.add_constant(author_stats['books_written'])
    y = author_stats['Rating']
    model = sm.OLS(y, X).fit()

    print(model.summary())

    plt.title('Author Rating vs. Number of Books Written')
    plt.xlabel('Number of Books Written')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.grid(True)
    plt.show()


plot_author_rating_vs_books_written()
