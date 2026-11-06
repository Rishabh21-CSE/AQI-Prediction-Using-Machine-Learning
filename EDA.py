import pandas as pd
import matplotlib.pyplot as plt

# Loading the dataset
df = pd.read_csv("city_day.csv")

# Removing rows where AQI is missing
df = df.dropna(subset=['AQI'])

# Checking basic AQI statistics
print(df['AQI'].describe())

# Plotting the distribution of AQI values
plt.figure(figsize=(8, 5))
plt.hist(df['AQI'], bins=30, edgecolor='black')
plt.xlabel("AQI")
plt.ylabel("Number of Days")
plt.title("Distribution of AQI")
plt.show()

# Finding the average AQI for each city
city_aqi = df.groupby('City')['AQI'].mean()

# Sorting cities by average AQI
city_aqi = city_aqi.sort_values(ascending=False)

# Displaying the top 10 cities
print(city_aqi.head(10))

# Plotting the average AQI by city
plt.figure(figsize=(10, 5))
city_aqi.head(10).plot(kind='bar')

plt.xlabel("City")
plt.ylabel("Average AQI")
plt.title("Top 10 Cities by Average AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

import seaborn as sns

# Selecting numerical columns for correlation
data = df[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
           'NH3', 'CO', 'SO2', 'O3', 'Benzene',
           'Toluene', 'Xylene', 'AQI']]

# Finding correlation between the columns
corr = data.corr()

# Plotting the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Correlation between Pollutants and AQI")
plt.tight_layout()
plt.show()

# Converting the Date column into date format
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

# Removing rows where Date or AQI is missing
data = df.dropna(subset=['Date', 'AQI'])

# Finding the average AQI for each date
daily_aqi = data.groupby('Date')['AQI'].mean()

# Sorting dates in order
daily_aqi = daily_aqi.sort_index()

# Plotting the AQI trend
plt.figure(figsize=(12, 5))
plt.plot(daily_aqi.index, daily_aqi.values)

plt.xlabel("Date")
plt.ylabel("Average AQI")
plt.title("AQI Trend Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Selecting pollutant columns
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
              'NH3', 'CO', 'SO2', 'O3', 'Benzene',
              'Toluene', 'Xylene']

# Calculating average pollutant levels
pollutant_avg = df[pollutants].mean()

# Displaying the average values
print(pollutant_avg)

# Plotting average pollutant levels
plt.figure(figsize=(10, 5))
pollutant_avg.plot(kind='bar')

plt.xlabel("Pollutants")
plt.ylabel("Average Concentration")
plt.title("Average Levels of Different Pollutants")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Finding useful information from the dataset

# Showing the top 5 cities by average AQI
city_aqi = df.groupby('City')['AQI'].mean()
city_aqi = city_aqi.sort_values(ascending=False)

print("\nTop 5 Cities by Average AQI:")
print(city_aqi.head())

# Finding the city with the lowest average AQI
print("\nCity with Lowest Average AQI:")
print(city_aqi.tail(1))

# Checking the correlation of pollutants with AQI
pollutants = ['PM2.5', 'PM10', 'NO', 'NO2', 'NOx',
              'NH3', 'CO', 'SO2', 'O3', 'AQI']

corr = df[pollutants].corr()

print("\nCorrelation of Pollutants with AQI:")
print(corr['AQI'].sort_values(ascending=False))

# Checking the date range of the dataset
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

print("\nFirst Date:", df['Date'].min())
print("Last Date:", df['Date'].max())

# Checking the average AQI
print("\nAverage AQI:", df['AQI'].mean())
print("Minimum AQI:", df['AQI'].min())
print("Maximum AQI:", df['AQI'].max())

# Checking unusually high AQI values
print("Highest AQI values:")
print(df.nlargest(10, 'AQI')[['City', 'Date', 'AQI']])

# Checking AQI values above 500
print("\nAQI values above 500:")
print(df[df['AQI'] > 500][['City', 'Date', 'AQI']])

# Counting AQI values above 500
print("\nNumber of AQI values above 500:")
print((df['AQI'] > 500).sum())