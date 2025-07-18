import pandas as pd
import random

def generate_values(label, count, condition):
    data = []
    for _ in range(count):
        row = {
            "Soil_Moisture": random.randint(*condition['Soil_Moisture']),
            "Temperature": round(random.uniform(*condition['Temperature']), 1),
            "Humidity": random.randint(*condition['Humidity']),
            "pH": round(random.uniform(*condition['pH']), 2),
            "Light": random.randint(*condition['Light']),
            "Rain_digital": random.randint(*condition['Rain_digital']),
            "Rain_analog": random.randint(*condition['Rain_analog']),
            "Best_Harvest": label
        }
        data.append(row)
    return data

# Conditions for each crop
wheat_condition = {
    "Soil_Moisture": (600, 750),
    "Temperature": (15, 25),
    "Humidity": (50, 70),
    "pH": (6.0, 7.5),
    "Light": (700, 1023),
    "Rain_digital": (0, 299),
    "Rain_analog": (0, 399)
}

rice_condition = {
    "Soil_Moisture": (750, 950),
    "Temperature": (25, 35),
    "Humidity": (70, 90),
    "pH": (5.5, 6.5),
    "Light": (600, 1023),
    "Rain_digital": (601, 1023),
    "Rain_analog": (501, 1023)
}

maize_condition = {
    "Soil_Moisture": (500, 700),
    "Temperature": (20, 30),
    "Humidity": (50, 70),
    "pH": (5.8, 7.0),
    "Light": (700, 1023),
    "Rain_digital": (0, 399),
    "Rain_analog": (0, 399)
}

# Generate data for each crop (equal count)
samples_per_crop = 666  # 666 * 3 ≈ 1998
dataset = []
dataset += generate_values("Wheat", samples_per_crop, wheat_condition)
dataset += generate_values("Rice", samples_per_crop, rice_condition)
dataset += generate_values("Maize", samples_per_crop, maize_condition)

# Add 2 extra rows to make total = 2000
dataset += generate_values("Wheat", 1, wheat_condition)
dataset += generate_values("Rice", 1, rice_condition)

# Create DataFrame and save
df = pd.DataFrame(dataset)
df.to_csv("crop_dataset.csv", index=False)

print("✅ crop_dataset.csv generated with 2000 rows.")
