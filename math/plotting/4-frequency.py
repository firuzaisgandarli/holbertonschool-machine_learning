#!/usr/bin/env python3
"""
Module that plots a histogram of student grades for Project A.
"""

import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student scores:
    - x-axis labeled 'Grades'
    - y-axis labeled 'Number of Students'
    - bins every 10 units
    - title 'Project A'
    - bars outlined in black
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))

    # Histogram with bins every 10 units and black outlines
    plt.hist(student_grades, bins=range(0, 101, 10), edgecolor='black')

    # Labels and title
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")

    # Optional: limit x-axis to 0–100 for cleaner binning
    plt.xlim(0, 100)

    # Show plot
    plt.show()
