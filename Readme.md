# Traffic Analysis for Brooklyn

## Overview
This project involves a detailed analysis of traffic accidents in Brooklyn, NYC, focusing on patterns of injuries and accident clustering based on data from June and July 2019 and 2020. The analysis utilizes Python with libraries like Pandas, GeoPandas, Folium, Matplotlib, Seaborn, and Scikit-Learn to manage, process, visualize, and analyze traffic data.

## Dataset
The data used in this analysis is sourced from a traffic.csv file, which includes details about each traffic incident within specified months in 2019 and 2020. Critical columns such as 'CRASH DATE', 'BOROUGH', 'NUMBER OF PERSONS INJURED', 'LATITUDE', and 'LONGITUDE' are utilized. To download the dataset go to the link:

```
https://drive.google.com/file/d/1HOgNp47ywjCPLsJZe1DObm64JH__ui7q/view?usp=sharing
```

## Dependencies
To run this project, you will need the following Python libraries:

Pandas
GeoPandas
Folium
Matplotlib
Seaborn
Scikit-Learn

You can install these packages using pip:
```
pip install pandas geopandas folium matplotlib seaborn scikit-learn
```
## Outputs
The script generates several outputs:

Heatmap: An HTML file (heatmap.html) visualizing the geographic distribution of accidents in Brooklyn.
Cluster Map: An HTML file (cluster_map_dbscan.html) displaying the results of DBSCAN clustering on accident locations.
Accident Time Analysis: Multiple PNG files such as Accidents_By_Hour_ofthe_day.png, Top_12_Days_2020_Line.png, and Accidents_By_Day_of_the_Week.png, showing various time-based analyses.
Correlation Plots and Distributions: A pair plot (Pair_Plot_Correlation.png) and histograms that analyze correlations and distributions of injuries and other factors.

## Key Findings

The most common hour and day for accidents.
Comparison of accident data between June and July for 2019 and 2020.
Identification of high-risk days and clustering of accident locations.
Impact of specific vehicle types (e.g., Amazon trucks) on traffic accidents.
For more details, refer to the output files generated by running the scripts.

## Contributing
Contributions to this project are welcome. You can contribute by:
Improving the efficiency of the scripts.
Extending the analysis to other boroughs or different time frames.
Enhancing the visualization features.
Please fork the repository and submit a pull request with your changes.
