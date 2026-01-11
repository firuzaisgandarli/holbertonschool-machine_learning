#!/usr/bin/env python3
"""
Module that plots a stacked bar graph of fruit per person.
"""

import numpy as np
import matplotlib.pyplot as plt


def bars():
    """
    Plot a stacked bar chart:
    - people: Farrah, Fred, Felicia
    - fruits (rows): apples, bananas, oranges, peaches
    - colors: apples=red, bananas=yellow, oranges=#ff8000, peaches=#ffe5b4
    - width=0.5, stacked in row order bottom→top
    - y-axis: 0–80, ticks every 10, label 'Quantity of Fruit'
    - title: 'Number of Fruit per Person'
    """
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    people = ['Farrah', 'Fred', 'Felicia']
    apples = fruit[0]
    bananas = fruit[1]
    oranges = fruit[2]
    peaches = fruit[3]

    x = np.arange(len(people))

    # Stacked bars in the order of rows: apples, bananas, oranges, peaches
    plt.bar(x, apples, width=0.5, color='red', label='apples')
    plt.bar(
        x, bananas, width=0.5, color='yellow',
        bottom=apples, label='bananas'
    )
    plt.bar(
        x, oranges, width=0.5, color='#ff8000',
        bottom=apples + bananas, label='oranges'
    )
    plt.bar(
        x, peaches, width=0.5, color='#ffe5b4',
        bottom=apples + bananas + oranges, label='peaches'
    )

    # X ticks per person
    plt.xticks(x, people)

    # Y-axis label, range, and ticks
    plt.ylabel("Quantity of Fruit")
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # Title and legend
    plt.title("Number of Fruit per Person")
    plt.legend()

    plt.show()
