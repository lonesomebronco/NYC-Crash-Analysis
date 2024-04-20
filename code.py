# Importing necessary libraries
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Setting environment variable for parallel computing
os.environ["OMP_NUM_THREADS"] = "1"

# Reading data from 'traffic.csv' file into a Pandas DataFrame
df = pd.read_csv('traffic.csv', dtype={'ZIP CODE': str})

# Defining critical columns and dropping rows with missing values in these columns
critical_columns = ['CRASH DATE', 'BOROUGH', 'NUMBER OF PERSONS INJURED', 'LATITUDE', 'LONGITUDE']
df = df.dropna(subset=critical_columns)

# Converting 'CRASH DATE' to datetime format
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'], errors='coerce')

# Converting 'NUMBER OF PERSONS INJURED' to integer type
df['NUMBER OF PERSONS INJURED'] = df['NUMBER OF PERSONS INJURED'].astype(int)

# Choosing a specific borough ('BROOKLYN') and time range (June and July of 2019 and 2020)
chosen_borough = 'BROOKLYN'
borough_data = df[(df['BOROUGH'] == chosen_borough) & ((df['CRASH DATE'].dt.year.isin([2019, 2020])) & (df['CRASH DATE'].dt.month.isin([6, 7])))]

# Creating a GeoDataFrame for the chosen borough data
gdf = gpd.GeoDataFrame(borough_data, geometry=gpd.points_from_xy(borough_data['LONGITUDE'], borough_data['LATITUDE']))

# Defining the center of the map
map_center = [borough_data['LATITUDE'].median(), borough_data['LONGITUDE'].median()]

# Initiating a Folium map
my_map = folium.Map(location=map_center, zoom_start=11)

# Creating a HeatMap layer on the map using latitude and longitude data
heat_data = [[point.xy[1][0], point.xy[0][0]] for point in gdf.geometry]
HeatMap(heat_data).add_to(my_map)

# Saving the map as an HTML file
my_map.save('heatmap.html')

# Initializing a matplotlib figure and axis for plotting
fig1, ax1 = plt.subplots(figsize=(10, 6))

# Creating a new column 'TOTAL PERSONS AFFECTED' by summing relevant columns
borough_data['TOTAL PERSONS AFFECTED'] = borough_data[['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
                                                       'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                                                       'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                                                       'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].sum(axis=1)

# Initializing DBSCAN clustering with specified parameters
eps = 0.01  
min_samples = 5  
dbscan = DBSCAN(eps=eps, min_samples=min_samples)

# Adding a 'cluster' column to GeoDataFrame based on DBSCAN clustering
gdf['cluster'] = dbscan.fit_predict(gdf[['LONGITUDE', 'LATITUDE']])

# Initiating a Folium map for cluster visualization
cluster_map = folium.Map(location=map_center, zoom_start=11)

# Iterating through clusters and adding markers to the cluster map
for cluster in gdf['cluster'].unique():
    cluster_data = gdf[gdf['cluster'] == cluster]
    for idx, row in cluster_data.iterrows():
        folium.Marker(location=[row['LATITUDE'], row['LONGITUDE']],
                      icon=folium.Icon(color=f'cluster{cluster + 1}'),
                      popup=f'Cluster: {cluster + 1}').add_to(cluster_map)

# Saving the cluster map as an HTML file
cluster_map.save('cluster_map_dbscan.html')


# PLOT 1 :Plotting a line plot over time for the total persons affected in accidents
sns.lineplot(x='CRASH DATE', y='TOTAL PERSONS AFFECTED', hue=borough_data['CRASH DATE'].dt.year, data=borough_data, ax=ax1, label='Total Persons Affected')

# Setting title and labels for the plot
ax1.set_title('Accidents Over Time (2019 vs. 2020)')
ax1.set_ylabel('Number of Persons Affected')
ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax1.legend() 
plt.tight_layout()
plt.savefig('Accidents_Over_Time.png')
# Showing the plot
plt.show()




# PLOT 2 :Initializing a new matplotlib figure and axis for plotting
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Creating a count plot for accidents by hour of the day
sns.countplot(x=borough_data['CRASH TIME'].apply(lambda x: int(x.split(':')[0])), data=borough_data, ax=ax3)

# Setting title and labels for the plot
ax3.set_title('Accidents by Hour of the Day')
ax3.set_xlabel('Hour of the Day')
plt.tight_layout()
plt.savefig('Accidents_By_Hour_ofthe_day.png')
# Showing the plot
plt.show()




# PLOT 3 :Initializing a new matplotlib figure and axis for plotting
fig3, ax3 = plt.subplots(figsize=(10, 6))

# Creating a new column 'Year' for the year of each accident
borough_data['Year'] = borough_data['CRASH DATE'].dt.year

# Plotting a count plot for accidents by hour of the day
sns.countplot(x=borough_data['CRASH TIME'].apply(lambda x: int(x.split(':')[0])), hue='Year', data=borough_data, ax=ax3)

# Setting title and labels for the plot
ax3.set_title('Accidents by Hour of the Day (2019 vs. 2020)')
ax3.set_xlabel('Hour of the Day')
plt.tight_layout()
plt.savefig('Accidents_By_Hour_ofthe_day.png')
# Showing the plot
plt.show()




# PLOT 4:Initializing a new matplotlib figure and axis for plotting
fig4, ax4 = plt.subplots(figsize=(12, 6))

# Creating a new column 'Month' for the month of each accident
borough_data['Month'] = borough_data['CRASH DATE'].dt.month_name()

# Plotting a violin plot for accidents by month
sns.violinplot(x='Month', y='NUMBER OF PERSONS INJURED', hue=borough_data['CRASH DATE'].dt.year, split=True, data=borough_data, ax=ax4)

# Setting title for the plot
ax4.set_title('Accidents by Month (2019 vs. 2020)')
plt.tight_layout()
plt.savefig('Accidents_By_Month.png')
# Showing the plot
plt.show()




# PLOT 5:Initializing a new matplotlib figure and axis for plotting
fig5, ax5 = plt.subplots(figsize=(10, 6))

# Updating 'TOTAL PERSONS AFFECTED' column with the sum of relevant columns
borough_data['TOTAL PERSONS AFFECTED'] = borough_data[['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED',
                                                       'NUMBER OF PEDESTRIANS INJURED', 'NUMBER OF PEDESTRIANS KILLED',
                                                       'NUMBER OF CYCLIST INJURED', 'NUMBER OF CYCLIST KILLED',
                                                       'NUMBER OF MOTORIST INJURED', 'NUMBER OF MOTORIST KILLED']].sum(axis=1)

# Selecting columns for pair plot
pair_plot_columns = ['NUMBER OF PERSONS INJURED', 'NUMBER OF PERSONS KILLED', 'NUMBER OF PEDESTRIANS INJURED',
                     'NUMBER OF CYCLIST INJURED', 'NUMBER OF MOTORIST INJURED']

# Creating a pair plot for correlation analysis
pair_plot_data = borough_data[pair_plot_columns]
sns.pairplot(pair_plot_data)

# Setting title for the pair plot
plt.suptitle('Pair Plot for Correlation', y=1.02)
plt.savefig('Pair_Plot_Correlation.png')
# Showing the pair plot
plt.show()




# PLOT 6: Selecting the top 12 days with the most accidents in 2020
top_12_days_2020 = borough_data[borough_data['CRASH DATE'].dt.year == 2020].nlargest(12, 'NUMBER OF PERSONS INJURED')['CRASH DATE']

# Initializing a new matplotlib figure and axis for plotting
fig_top_12_days, ax_top_12_days = plt.subplots(figsize=(12, 6))

# Plotting the line plot for the top 12 days in 2020
sns.lineplot(x=top_12_days_2020, y=range(1, 13), marker='o', markersize=8, color='green', ax=ax_top_12_days)

# Setting title and labels for the plot
ax_top_12_days.set_title('Top 12 Days with Most Accidents in 2020')
ax_top_12_days.set_xlabel('Date')
ax_top_12_days.set_ylabel('Ranking')
ax_top_12_days.invert_yaxis() 
plt.tight_layout()
plt.savefig('Top_12_Days_2020_Line.png')
# Showing the plot
plt.show()





# PLOT 7: Plotting a bar chart for accidents by day of the week
df['CRASH DATE'] = pd.to_datetime(df['CRASH DATE'])
weekday_counts = df['CRASH DATE'].dt.day_name().value_counts()

# Initializing a new matplotlib figure and axis for plotting
fig_weekday, ax_weekday = plt.subplots(figsize=(10, 6))

# Plotting a bar chart for accidents by day of the week
sns.barplot(x=weekday_counts.index, y=weekday_counts.values, palette='viridis', ax=ax_weekday)

# Setting title and labels for the plot
ax_weekday.set_title('Accidents by Day of the Week')
ax_weekday.set_xlabel('Day of the Week')
ax_weekday.set_ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Accidents_By_Day_of_the_Week.png')
# Showing the plot
plt.show()



# Extracting data for June 2019 and June 2020
june_2019 = borough_data[(borough_data['CRASH DATE'].dt.year == 2019) & (borough_data['CRASH DATE'].dt.month == 6)]
june_2020 = borough_data[(borough_data['CRASH DATE'].dt.year == 2020) & (borough_data['CRASH DATE'].dt.month == 6)]

# Calculating the difference in the number of persons injured between June 2019 and June 2020
difference_june = june_2020['NUMBER OF PERSONS INJURED'].sum() - june_2019['NUMBER OF PERSONS INJURED'].sum()

# Extracting data for July 2019 and July 2020
july_2019 = borough_data[(borough_data['CRASH DATE'].dt.year == 2019) & (borough_data['CRASH DATE'].dt.month == 7)]
july_2020 = borough_data[(borough_data['CRASH DATE'].dt.year == 2020) & (borough_data['CRASH DATE'].dt.month == 7)]

# Calculating the difference in the number of persons injured between July 2019 and July 2020
difference_july = july_2020['NUMBER OF PERSONS INJURED'].sum() - july_2019['NUMBER OF PERSONS INJURED'].sum()

# Converting 'CRASH DATE' column to datetime format
borough_data.loc[:, 'CRASH DATE'] = pd.to_datetime(borough_data['CRASH DATE'])

# Calculating the rolling sum of 'NUMBER OF PERSONS INJURED' over a window of 100 days
rolling_sum = borough_data.set_index('CRASH DATE')['NUMBER OF PERSONS INJURED'].rolling(window=100).sum()

# Selecting the top 100 days with the most accidents
top_100_days = rolling_sum.nlargest(100)

# Counting the number of accidents for each day of the week
day_of_week_counts = borough_data['CRASH DATE'].dt.day_name().value_counts()

# Checking if day_of_week_counts is not empty before finding the day with the most accidents
if not day_of_week_counts.empty:
    most_accidents_day = day_of_week_counts.idxmax()
    print("5. Day of the Week with Most Accidents:", most_accidents_day)
else:
    print("5. No data available for the chosen borough and time range.")

# Finding the hour of the day with the most accidents
hour_of_day_counts = borough_data['CRASH DATE'].dt.hour.value_counts()

most_accidents_hour = hour_of_day_counts.idxmax()

# Selecting the top 12 days with the most accidents in 2020
top_12_days_2020 = borough_data[borough_data['CRASH DATE'].dt.year == 2020].nlargest(12, 'NUMBER OF PERSONS INJURED')['CRASH DATE']

# Printing analysis results
print("1. Changes in Summer from One Year to Next (Visualization Shown)")
print("2. Difference in June 2019 and June 2020:", difference_june)
print("3. Difference in July 2019 and July 2020:", difference_july)
print("4. Top 100 Consecutive Days with Most Accidents:")
print(top_100_days)
hour_of_day_counts = borough_data['CRASH DATE'].dt.hour.value_counts()
most_accidents_hour = hour_of_day_counts.idxmax()

# Convert to 12-hour format with AM/PM designation
most_accidents_hour_12hr = most_accidents_hour % 12 if most_accidents_hour % 12 != 0 else 12
ampm = 'AM' if most_accidents_hour < 12 else 'PM'

# Finding the hour of the day with the most accidents in 2019
hour_of_day_counts_2019 = borough_data[borough_data['CRASH DATE'].dt.year == 2019]['CRASH TIME'].apply(lambda x: int(x.split(':')[0])).value_counts()
most_accidents_hour_2019 = hour_of_day_counts_2019.idxmax()

# Finding the hour of the day with the most accidents in 2020
hour_of_day_counts_2020 = borough_data[borough_data['CRASH DATE'].dt.year == 2020]['CRASH TIME'].apply(lambda x: int(x.split(':')[0])).value_counts()
most_accidents_hour_2020 = hour_of_day_counts_2020.idxmax()

print("6. Most accidents in 2019 occurred at:", f"{most_accidents_hour_2019}:00 hours")

print("7. 12 Days with Most Accidents in 2020 (with Speculation):")
print(top_12_days_2020)

# Finding the day of the week most common for accidents
most_common_day = df['CRASH DATE'].dt.day_name().mode().iloc[0]
print("8. Day of the Week Most Common for Accidents:", most_common_day)

print("9. Most accidents in 2020 occurred at:", f"{most_accidents_hour_2020}:00 hours")
# Finding the day of the year most common for accidents
most_common_day_of_year = df['CRASH DATE'].dt.strftime('%d/%m').mode().iloc[0]
print("10. Day of the Year Most Common (dd/mm):", most_common_day_of_year)

# Finding the date of the month most common for accidents
most_common_date_of_month = df['CRASH DATE'].dt.strftime('%d').mode().iloc[0]
print("11. Date of the Month Most Common:", most_common_date_of_month)


# Extracting data for crashes involving Amazon trucks
amazon_truck_crashes = df[df['VEHICLE TYPE CODE 1'].str.contains('AMAZON', case=False) |
                          df['VEHICLE TYPE CODE 2'].str.contains('AMAZON', case=False) |
                          df['VEHICLE TYPE CODE 3'].str.contains('AMAZON', case=False) |
                          df['VEHICLE TYPE CODE 4'].str.contains('AMAZON', case=False) |
                          df['VEHICLE TYPE CODE 5'].str.contains('AMAZON', case=False)]

# Checking if Amazon truck crashes are an issue based on the number of incidents
are_amazon_truck_crashes_an_issue = len(amazon_truck_crashes) > 0
print("12. Are Amazon Truck Crashes an Issue?", are_amazon_truck_crashes_an_issue)



# PLOT 8: Plotting the most common type of crash
plt.figure(figsize=(10, 6))
sns.countplot(x='CONTRIBUTING FACTOR VEHICLE 1', data=df, order=df['CONTRIBUTING FACTOR VEHICLE 1'].value_counts().index)
plt.title('Most Common Type of Crash')
plt.xlabel('Type of Crash')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('Most_Comon_type_of_crash.png')
plt.show()