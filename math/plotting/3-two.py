#!/usr/bin/env python3
"""
Module that plots exponential decay of two radioactive elements:
C-14 and Ra-226, with specified styles and legend.
"""

import numpy as np
import matplotlib.pyplot as plt


def two():
    """
    Plot x ↦ y1 (C-14) and x ↦ y2 (Ra-226) as line graphs with:
    - x-axis labeled 'Time (years)'
    - y-axis labeled 'Fraction Remaining'
    - title 'Exponential Decay of Radioactive Elements'
    - x-axis range [0, 20000]
    - y-axis range [0, 1]
    - y1 dashed red line, y2 solid green line
    - legend in upper right
    """
    x = np.arange(0, 21000, 1000)
    r = np.log(0.5)
    t1 = 5730
    t2 = 1600
    y1 = np.exp((r / t1) * x)
    y2 = np.exp((r / t2) * x)
    plt.figure(figsize=(6.4, 4.8))

    # Plot lines with required styles
    plt.plot(x, y1, 'r--', label='C-14')
    plt.plot(x, y2, 'g-', label='Ra-226')

    # Labels and title
    plt.xlabel("Time (years)")
    plt.ylabel("Fraction Remaining")
    plt.title("Exponential Decay of Radioactive Elements")

    # Axis ranges
    plt.xlim(0, 20000)
    plt.ylim(0, 1)

    # Legend in upper right
    plt.legend(loc='upper right')

    # Show plot
    plt.show()
