# Import libraries
import pandas as pd
import numpy as np
from ipums_lib import row_generator, columm_generator

pd.set_option('display.max_columns', None)


# ------------------------------------------------------------------------------------------------------------------- #
#                                                 Importing Dataset                                                   #
# ------------------------------------------------------------------------------------------------------------------- #
# Average city temperatures (FSR): https://www.fetchseries.com/climate/average-city-temperatures-fsr/
temperature_excel_filename = r"Source_Data_Files/average-city-temperatures-fsr.xlsx"

# USA Real Estate Dataset: https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset/data
realtor_data = r"Source_Data_Files/realtor-data.csv"

# Current Population Survey (CPS) Data
cps_data = r"Source_Data_Files/cps_00001.dat"
cps_ddi = r"Source_Data_Files/cps_00001.xml"

# ------------------------------------------------------------------------------------------------------------------- #
#                                       Data Preparation - Average Temperatures                                       #
# ------------------------------------------------------------------------------------------------------------------- #
def prep_temperature_data(temperature_source_file):
    temperature_data = pd.read_excel(temperature_source_file)

    # Set first row as header
    temperature_data.columns = temperature_data.iloc[0]
    temperature_data = temperature_data[1:]

    # Get all columns containing 'United States'
    usa_cols = temperature_data.filter(like='United States', axis=1).columns
    valid_columns = [col for col in usa_cols if col in temperature_data.columns]

    # Unpivot (melt) the DataFrame
    avg_us_temp_df = pd.melt(
        temperature_data,
        id_vars=['Date string'],            # Columns to keep as identifiers
        value_vars=valid_columns,           # Unpivot all US city columns
        var_name='city',                    # Name for the new 'variable' column
        value_name='average_temp_celsius'           # Name for the new 'value' column
    )

    #print("\nAverage US Temperature Data:")
    #print(avg_us_temp_df)

    substring_to_remove = "United States-"
    avg_us_temp_df['city'] = avg_us_temp_df['city'].str.replace(substring_to_remove, '', regex=False)

    avg_us_temp_df.to_csv("avg_us_temp_celsius.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------- #
#                                            Data Preparation - Real Estate                                           #
# ------------------------------------------------------------------------------------------------------------------- #
#realtor_df = pd.read_csv(realtor_data)
# NEED TO FIX: prev_sold_date has mixed types

# ------------------------------------------------------------------------------------------------------------------- #
#                                        Data Preparation - Current Population                                        #
# ------------------------------------------------------------------------------------------------------------------- #
# Get column names and initialize dataframe
columns = columm_generator(datapath = cps_data, ddipath = cps_ddi)
columns = [column for column in columns]
cps_df = pd.DataFrame(columns=columns)

# Get rows from row generator and append to dataframe
rows = row_generator(datapath = cps_data, ddipath = cps_ddi)
for row in rows:
    cps_df.loc[len(cps_df)] = row

print(cps_df)
cps_df.to_csv("CSV_Outputs/cps_data.csv", index=False)

# ------------------------------------------------------------------------------------------------------------------- #
#                                                         Main                                                        #
# ------------------------------------------------------------------------------------------------------------------- #
#prep_temperature_data(temperature_excel_filename)
#temperature_df = pd.read_csv("CSV_Outputs/avg_us_temp_celsius.csv")

#merged_df = pd.merge(realtor_df, temperature_df, on='city', how='inner')
#merged_df.to_csv("CSV_Outputs/temp_realtor_data.csv", index=False)
#print(merged_df)

#merged_df = pd.read_csv("temp_realtor_data.csv")

# Result: -0.010921317058140189
#corr = merged_df['price'].corr(merged_df['average_temp_celsius'])
#print(corr)
