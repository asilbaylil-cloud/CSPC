import pandas as pd
import os

results = pd.read_csv("PW5/results/events.csv")

report = ""

for i, row in results.iterrows():
    report += f"Earthquake {row['Earthquake']}\n"
    report += f"P-wave arrival: {row['P_arrival']:.2f} s\n"
    report += f"S-wave arrival: {row['S_arrival']:.2f} s\n"
    report += f"Distance: {row['Distance_km']:.2f} km\n"
    report += f"Magnitude: {row['Magnitude']:.2f}\n\n"

closest = results.loc[results["Distance_km"].idxmin()]
strongest = results.loc[results["Magnitude"].idxmax()]

report += f"Earthquake {closest['Earthquake']} was closer.\n"
report += f"Earthquake {strongest['Earthquake']} had the higher magnitude.\n"

os.makedirs("PW5/results", exist_ok=True)

with open("PW5/results/report.txt", "w") as file:
    file.write(report)

print(report)