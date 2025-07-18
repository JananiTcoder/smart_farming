import pandas as pd
import numpy as np
import random

# Seed for reproducibility
random.seed(42)
np.random.seed(42)

# Function to generate a single sensor row for a given crop and price level
def generate_crop_row(crop, price_level):
    if crop == "wheat":
        if price_level == "high":
            sm = np.random.randint(600, 751)
            temp = np.random.uniform(15, 25)
            ph = np.random.uniform(6.0, 7.5)
            hum = np.random.uniform(50, 70)
            light = np.random.randint(701, 1024)
        elif price_level == "medium":
            sm = np.random.randint(550, 800)
            temp = np.random.uniform(25, 30)
            ph = np.random.uniform(5.5, 6.0)
            hum = np.random.uniform(50, 80)
            light = np.random.randint(600, 1024)
        else:  # low
            sm = np.random.randint(300, 500)
            temp = np.random.uniform(32, 40)
            ph = np.random.uniform(4.5, 8.5)
            hum = np.random.uniform(30, 90)
            light = np.random.randint(300, 900)

    elif crop == "rice":
        if price_level == "high":
            sm = np.random.randint(750, 951)
            temp = np.random.uniform(25, 35)
            ph = np.random.uniform(5.5, 6.5)
            hum = np.random.uniform(71, 90)
            light = np.random.randint(600, 1024)
        elif price_level == "medium":
            sm = np.random.randint(600, 749)
            temp = np.random.uniform(20, 38)
            ph = np.random.uniform(5.0, 7.0)
            hum = np.random.uniform(60, 80)
            light = np.random.randint(500, 1024)
        else:  # low
            sm = np.random.randint(300, 599)
            temp = np.random.uniform(10, 20)
            ph = np.random.uniform(7.6, 8.5)
            hum = np.random.uniform(40, 70)
            light = np.random.randint(300, 800)

    elif crop == "maize":
        if price_level == "high":
            sm = np.random.randint(500, 701)
            temp = np.random.uniform(20, 30)
            ph = np.random.uniform(5.8, 7.0)
            hum = np.random.uniform(50, 70)
            light = np.random.randint(701, 1024)
        elif price_level == "medium":
            sm = np.random.randint(450, 750)
            temp = np.random.uniform(18, 35)
            ph = np.random.uniform(5.5, 7.5)
            hum = np.random.uniform(40, 80)
            light = np.random.randint(600, 1024)
        else:  # low
            sm = np.random.randint(300, 399)
            temp = np.random.uniform(35, 40)
            ph = np.random.uniform(4.5, 5.4)
            hum = np.random.uniform(30, 90)
            light = np.random.randint(300, 700)

    rain_digital = np.random.randint(0, 1024)
    rain_analog = np.random.randint(0, 1024)

    return {
        "Soil_Moisture": sm,
        "Temperature": round(temp, 2),
        "Humidity": round(hum, 2),
        "pH": round(ph, 2),
        "Light": light,
        "Rain_digital": rain_digital,
        "Rain_analog": rain_analog,
        "crop_type": crop,
        "cost": price_level
    }

# Generate equal rows for each cost level (666 each for low, medium, high)
rows_per_label = 667  # 667 * 3 ≈ 2000
crops = ["wheat", "rice", "maize"]
price_levels = ["low", "medium", "high"]

data = []
for price in price_levels:
    for _ in range(rows_per_label):
        crop = random.choice(crops)
        data.append(generate_crop_row(crop, price))

# Save to CSV
df_price = pd.DataFrame(data)
csv_price_path = "/mnt/data/smart_crop_price_dataset.csv"
df_price.to_csv(csv_price_path, index=False)

csv_price_path
