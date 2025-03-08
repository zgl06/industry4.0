import pandas as pd
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv("tmc_most_recent_summary_data.csv")

print(df.head())
print(df.columns.str.strip().str.lower())


plt.figure(figsize=(10, 6))
plt.scatter(df["longitude"], df["latitude"], c="red", alpha=0.5)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Location Visualization")
plt.grid(True)
plt.show()
