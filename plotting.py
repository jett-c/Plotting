import matplotlib.pyplot as plt
import numpy as np

def linear_model(B, x):
    return B[0] * x + B[1]

# Sample data: replace these with your actual data
x = np.array([5.1958E+14, 5.4908E+14, 6.8761E+14, 7.4025E+14, 8.2137E+14])  # x-coordinates
x_err = np.zeros_like(x) # x-errors
y_1 = np.array([0.501, 0.592, 1.136, 1.363, 1.812])  # y-coordinates
y_2 = np.array([0.479, 0.589, 1.13, 1.35, 1.791])
y_3 = np.array([0.48, 0.589, 1.13, 1.344, 1.778])
y_err = np.array([0.001, 0.001, 0.001, 0.001, 0.001]) # y-errors
x_label = r'$\nu/$Hz'  # x-axis label
y_label = r'$V_{\text{stop}}/V$'  # y-axis label
title = 'Stopping Potential vs Incident Light Frequency'  # plot title

# Create the plot
plt.figure(figsize=(8, 6))

# Function to perform linear regression and plot
def perform_lsa(x, y, y_err, label, color):
    plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt='o', capsize=2, markersize=5, label=f'{label}', color=color)

    # Perform linear least squares fitting
    p, cov = np.polyfit(x, y, 1, cov=True)
    slope, intercept = p
    slope_err, intercept_err = np.sqrt(np.diag(cov))
    print(f"Aperture Diameter {label}:")
    print(f"Slope {label}:", slope, "Intercept:", intercept)
    print(f"Slope error {label}:", slope_err, "Intercept error:", intercept_err)
    print(f'{y_label} = ({slope:.2e} ± {slope_err:.2e}) ({x_label}) {intercept:+.2e} ± {intercept_err:+.2e}')

    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = linear_model([slope, intercept], x_fit)

    # Predicted values
    y_pred = linear_model([slope, intercept], x)
    y_mean = np.mean(y)
    tss = np.sum((y - y_mean)**2)
    rss = np.sum((y - y_pred)**2)
    r_squared = 1 - (rss / tss)

    print(f'R² {label} = {r_squared:.5f}')

    # Plot the fit line
    plt.plot(x_fit, y_fit, linestyle='--', linewidth=2, color=color, alpha=0.5)

# Perform LSA on each pair of data with specified colors
perform_lsa(x, y_1, y_err, '2 mm', 'blue')
perform_lsa(x, y_2, y_err, '4 mm', 'green')
perform_lsa(x, y_3, y_err, '8 mm', 'red')

# Add titles and labels
plt.title(title)
plt.xlabel(x_label)
plt.ylabel(y_label)

# Show grid
plt.grid(True)

# Add legend
plt.legend()

# Display the plot
plt.show()
