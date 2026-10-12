"""
Task 3: Logistics Data Analysis Using Python
Install: pip install pandas numpy matplotlib
Run: python logistics_analysis.py
The dataset is simulated and intended for educational analysis.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "Shipment_ID": range(1, n + 1),
    "Distance_km": np.random.randint(10, 1000, n),
    "Shipment_Weight_kg": np.random.randint(5, 500, n),
    "Shipment_Volume": np.random.randint(1, 100, n),
    "Transport_Mode": np.random.choice(["Road", "Rail", "Air", "Sea"], n),
    "Delivery_Time_hours": np.random.randint(2, 120, n),
    "Transportation_Cost": np.random.randint(500, 20000, n),
    "Delivery_Status": np.random.choice(["On Time", "Delayed"], n, p=[0.75, 0.25])
})

os.makedirs("data", exist_ok=True)
os.makedirs("visualizations", exist_ok=True)
df.to_csv("data/logistics_dataset.csv", index=False)

numeric = [
    "Distance_km", "Shipment_Weight_kg", "Shipment_Volume",
    "Delivery_Time_hours", "Transportation_Cost"
]

print("Dataset shape:", df.shape)
print("\nDataset information:")
df.info()
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nDescriptive statistics:\n", df[numeric].describe().round(2))
print("\nMeans:\n", df[numeric].mean().round(2))
print("\nMedians:\n", df[numeric].median().round(2))
print("\nModes:\n", df[numeric].mode().iloc[0])
print("\nDelivery status counts:\n", df["Delivery_Status"].value_counts())

mode_summary = df.groupby("Transport_Mode").agg(
    Shipment_Count=("Shipment_ID", "count"),
    Average_Distance=("Distance_km", "mean"),
    Average_Delivery_Time=("Delivery_Time_hours", "mean"),
    Average_Cost=("Transportation_Cost", "mean")
).round(2)
print("\nPerformance by transport mode:\n", mode_summary)

corr = df[numeric].corr()
print("\nCorrelation matrix:\n", corr.round(2))

# Chart 1: Histogram - delivery-time distribution
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.hist(df["Delivery_Time_hours"], bins=20, edgecolor="black")
ax.set(title="Distribution of Delivery Times",
       xlabel="Delivery Time (Hours)", ylabel="Number of Shipments")
fig.tight_layout()
fig.savefig("visualizations/01_delivery_time_distribution.png", dpi=220)
plt.close(fig)

# Chart 2: Bar chart - delivery status counts
counts = df["Delivery_Status"].value_counts().reindex(["On Time", "Delayed"])
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(counts.index, counts.values, edgecolor="black")
ax.set(title="On-Time vs Delayed Shipments",
       xlabel="Delivery Status", ylabel="Number of Shipments")
fig.tight_layout()
fig.savefig("visualizations/02_delivery_status.png", dpi=220)
plt.close(fig)

# Chart 3: Bar chart - mean transportation cost by mode
costs = df.groupby("Transport_Mode")["Transportation_Cost"].mean().reindex(
    ["Road", "Air", "Rail", "Sea"])
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(costs.index, costs.values, edgecolor="black")
ax.set(title="Average Transportation Cost by Mode",
       xlabel="Transportation Mode", ylabel="Average Cost (Cost Units)")
fig.tight_layout()
fig.savefig("visualizations/03_cost_by_transport_mode.png", dpi=220)
plt.close(fig)

# Chart 4: Scatter plot - distance versus delivery time
fig, ax = plt.subplots(figsize=(8, 4.6))
for status, marker in [("On Time", "o"), ("Delayed", "x")]:
    part = df[df["Delivery_Status"] == status]
    ax.scatter(part["Distance_km"], part["Delivery_Time_hours"],
               label=status, marker=marker, alpha=0.65)
ax.set(title="Distance vs Delivery Time",
       xlabel="Distance (km)", ylabel="Delivery Time (Hours)")
ax.legend()
fig.tight_layout()
fig.savefig("visualizations/04_distance_vs_delivery_time.png", dpi=220)
plt.close(fig)

# Chart 5: Correlation heatmap
fig, ax = plt.subplots(figsize=(8, 5.5))
im = ax.imshow(corr.values, vmin=-1, vmax=1, aspect="auto")
labels = [c.replace("_", "\n") for c in numeric]
ax.set_xticks(range(len(numeric)))
ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(len(numeric)))
ax.set_yticklabels([c.replace("_", " ") for c in numeric], fontsize=8)
for i in range(len(numeric)):
    for j in range(len(numeric)):
        ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                ha="center", va="center", fontsize=8)
fig.colorbar(im, ax=ax, label="Pearson correlation")
ax.set_title("Correlation Between Numerical Variables")
fig.tight_layout()
fig.savefig("visualizations/05_correlation_heatmap.png", dpi=220)
plt.close(fig)

# Chart 6: Box plot - delivery-time distribution by mode
modes = ["Road", "Air", "Rail", "Sea"]
box_data = [df.loc[df["Transport_Mode"] == m, "Delivery_Time_hours"]
            for m in modes]
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.boxplot(box_data, labels=modes)
ax.set(title="Delivery Time Distribution by Transport Mode",
       xlabel="Transportation Mode", ylabel="Delivery Time (Hours)")
fig.tight_layout()
fig.savefig("visualizations/06_delivery_time_by_mode.png", dpi=220)
plt.close(fig)

print("\nCompleted: dataset, EDA, and six visualizations saved.")
