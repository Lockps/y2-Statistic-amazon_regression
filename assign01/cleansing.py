import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import statsmodels.api as sm

df = pd.read_csv('./data/amazon.csv')


def plot_author_rating_vs_books_written():
    author_stats = df.groupby('author').agg(
        {'stars': 'mean', 'asin': 'count'}).reset_index()
    author_stats = author_stats.rename(columns={'asin': 'books_written'})
    author_stats = author_stats.sort_values(by='books_written')

    plt.figure(figsize=(10, 6))
    plt.scatter(author_stats['books_written'],
                author_stats['stars'], alpha=0.5, label='Data Points')

    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    popt, pcov = curve_fit(
        func, author_stats['books_written'], author_stats['stars'])

    x_values = np.linspace(0, author_stats['books_written'].max(), 100)
    y_values = func(x_values, *popt)
    plt.plot(x_values, y_values, color='red', label='Curve Fit')

    plt.title('Author Rating vs. Number of Books Written')
    plt.xlabel('Number of Books Written')
    plt.ylabel('Average Rating')
    plt.legend()
    plt.grid(True)
    plt.show()

    # OLS regression
    X = sm.add_constant(author_stats['books_written'])
    y = author_stats['stars']
    model = sm.OLS(y, X).fit()

    # Print regression results summary
    print(model.summary())


plot_author_rating_vs_books_written()
