import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

from Point import Point


def run1():
    def format_float(num, formatted=True, digit=2):
        if formatted:
            return round(num, digit)
        else:
            return num

    x = [2, 4, 1, 5, 0, 8, 8.5, 2, 4.5, 10]
    y = [3, 4, 2.5, 6, 1, 6, 7, 2, 5.5, 10]
    x2 = [i ** 2 for i in x]
    y2 = [i ** 2 for i in y]
    xy = [x * y for (x, y) in zip(x, y)]
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x2 = sum(x2)
    sum_y2 = sum(y2)
    sum_xy = sum(xy)
    N = len(x)
    w1 = (N * sum_xy - sum_x * sum_y) / (N * sum_x2 - sum_x ** 2)
    w0 = (1 / N) * (sum_y - w1 * sum_x)
    print("x:", x, "sum of x:", sum_x)
    print("y:", y, "sum of y:", sum_y)
    print("x^2:", x2, "sum of x^2:", sum_x2)
    print("y^2:", y2, "sum of y^2:", sum_y2)
    print("xy:", xy, "sum of xy:", sum_xy)
    print("N:", N)
    print("w1: (N * sum of xy - sum of x * sum of y) / (N * sum of x'2 - (sum of x)^2)")
    print("w0: (1 / N) * (sum of y - w1 * sum of x)")
    print("w1:", format_float(w1))
    print("w0:", format_float(w0))
    print("f(x) = w1x + w0")
    print(f"f(x) = {format_float(w1)}x + {format_float(w0)}")

    f = lambda a: w1 * a + w0
    y_pred = [f(i) for i in x]
    print("y_pred:", end=" ")
    [print(format_float(i), end=", ") for i in y_pred]
    print()
    errors = [(yi - y_p) ** 2 for yi, y_p in zip(y, y_pred)]
    print("errors:", end=" ")
    [print(format_float(i), end=", ") for i in errors]
    print()
    error = sum(errors)
    print("ERROR:", format_float(error))

    plt.plot(x, y, 'ro')
    plt.plot(x, y_pred)
    plt.show()


def run2():
    points = [tuple(Point(*line.strip().split('\t'))) for line in open("data/points.txt")]
    print(points)

    df = pd.DataFrame(points, columns=['x', 'y'])
    k_means = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(df)

    sns.scatterplot(data=df, x="x", y="y", palette="tab10", hue=k_means.labels_, legend=False)
    plt.scatter(k_means.cluster_centers_[:, 0], k_means.cluster_centers_[:, 1],
                marker="X", c="black", s=80, label="centroids")
    plt.legend(loc=0, fontsize="small")
    plt.show()


if __name__ == '__main__':
    run1()
    run2()
