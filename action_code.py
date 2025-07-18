import pandas as pd
import numpy as np
import random

# Random seed for reproducibility
random.seed(42)
np.random.seed(42)

def add_noise(value, percent=10):
    noise = value * (np.random.uniform(-percent/100, percent/100))
    return max(0, min(value + noise, 1023))  # keep within sensor range

def generate_data(num_samples_per_class=700):
    data = []
    actions = ["Irrigate", "Fertilize", "Spray Pesticide"]
    crops = ["wheat", "rice", "maize"]

    for action in actions:
        for _ in range(num_samples_per_class):
            crop_type = random.choice(crops)

            # base values
            if action == "Irrigate":
                sm = np.random.randint(300, 500)
                temp = np.random.uniform(20, 35)
                hum = np.random.uniform(30, 60)
                ph = np.random.uniform(6.2, 7.5)
                light = np.random.randint(0, 400)
                rain_digital = np.random.randint(800, 1023)
                rain_analog = np.random.randint(700, 1023)

            elif action == "Fertilize":
                sm = np.random.randint(500, 750)
                temp = np.random.uniform(20, 30)
                hum = np.random.uniform(40, 70)
                ph = np.random.uniform(6.0, 7.5)
                light = np.random.randint(400, 800)
                rain_digital = np.random.randint(700, 1023)
                rain_analog = np.random.randint(400, 750)

            elif action == "Spray Pesticide":
                sm = np.random.randint(500, 850)
                temp = np.random.uniform(15, 30)
                hum = np.random.uniform(50, 85)
                ph = np.random.uniform(5.5, 7.0)
                light = np.random.randint(0, 300)
                rain_digital = np.random.randint(700, 1023)
                rain_analog = np.random.randint(600, 1023)

            # Apply ±5–10% noise
            sm = int(add_noise(sm, percent=10))
            temp = round(add_noise(temp, percent=5), 2)
            hum = round(add_noise(hum, percent=5), 2)
            ph = round(min(9.0, max(3.5, add_noise(ph, percent=5))), 2)
            light = int(add_noise(light, percent=10))
            rain_digital = int(add_noise(rain_digital, percent=10))
            rain_analog = int(add_noise(rain_analog, percent=10))

            data.append([sm, temp, hum, ph, light, rain_digital, rain_analog, crop_type, action])

    return pd.DataFrame(data, columns=[
        "Soil_Moisture", "Temperature", "Humidity", "pH", "Light",
        "Rain_digital", "Rain_analog", "crop_type", "action"
    ])

# Generate data
df = generate_data(num_samples_per_class=666)  # ~2000 rows total
df.to_csv("action_noisy_realistic.csv", index=False)
print("✅ Dataset created and saved as 'action_noisy_realistic.csv'")
