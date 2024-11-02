# utils.py

import matplotlib.pyplot as plt

def plot_transformation(T, v1, v2):
    # Dapatkan hasil transformasi
    w1 = T(v1)
    w2 = T(v2)

    # Plot vektor asli
    plt.quiver(0, 0, v1[0, 0], v1[1, 0], angles='xy', scale_units='xy', scale=1, color='blue', label='e1 (original)')
    plt.quiver(0, 0, v2[0, 0], v2[1, 0], angles='xy', scale_units='xy', scale=1, color='red', label='e2 (original)')

    # Plot vektor setelah transformasi
    plt.quiver(0, 0, w1[0, 0], w1[1, 0], angles='xy', scale_units='xy', scale=1, color='cyan', label='e1 (transformed)')
    plt.quiver(0, 0, w2[0, 0], w2[1, 0], angles='xy', scale_units='xy', scale=1, color='orange', label='e2 (transformed)')

    plt.xlim(-3, 3)
    plt.ylim(-3, 3)
    plt.axhline(0, color='black', lw=0.5)
    plt.axvline(0, color='black', lw=0.5)
    plt.grid()
    plt.legend()
    plt.title("Linear Transformation Visualization")
    plt.show()
