import geopandas as gpd
import pandas as pd
import censusdis.data as ced
from censusdis.data import variables
from censusdis.datasets import ACS1
from censusdis import states
from censusdis.states import ALL_STATES_AND_DC
from censusdis.states import NY
from censusdis.counties.new_york import QUEENS
from censusdis.multiyear import download_multiyear

pd.set_option('display.max_columns',None)

# Source: https://github.com/censusdis/censusdis/blob/main/notebooks/Column%20Labels.ipynb
def name_mapper(variable: str):
    """Map from the variables we got back to their labels."""
    if variable.startswith(GROUP):
        # Look up details of the particular variable:
        vars = ced.variables.search(ACS1, VINTAGE, group_name=GROUP, name=variable)
        # Get the label and parse out the part we want:
        label = vars.iloc[0]["LABEL"]
        return label.split("!")[-1].split(":")[0]
    else:
        # Not in the group we are interested in, so leave it as is.
        return variable


# Label descriptions for variable names: https://api.census.gov/data/2024/acs/acs1/variables.html
TOTAL_POPULATION = "B01003_001E"
MEDIAN_GROSS_RENT = "B25064_001E"


median_home_value_by_race_vars = {
    "B25077_001E": "MEDIAN_HOME_VALUE",
    #"B25077A_001E": "MEDIAN_HOME_VALUE_WHITE",
    #"B25077B_001E": "MEDIAN_HOME_VALUE_BLACK",
    #"B25077C_001E": "MEDIAN_HOME_VALUE_NATIVE",
    #"B25077D_001E": "MEDIAN_HOME_VALUE_ASIAN",
    #"B25077E_001E": "MEDIAN_HOME_VALUE_PACIFIC",
    #"B25077F_001E": "MEDIAN_HOME_VALUE_OTHER",
    #"B25077G_001E": "MEDIAN_HOME_VALUE_MIXED",
    #"B25077I_001E": "MEDIAN_HOME_VALUE_HISPANIC"
}

ECON_VARS = {
    "B19013_001E": "MEDIAN_HOUSEHOLD_INCOME",
    "C25004_002E": "VACANT_HOUSING"
}


# Combine all into one list for the API call
#all_variables = list(total_by_race_vars.keys()) + list(ECON_VARS.keys())

# Download data
#df = ced.download(
    # Data set: American Community Survey 1-Year - https://www.census.gov/data/developers/data-sets/acs-1year.html
    #dataset=ACS1,

    # Year
    #vintage=2024,

    # Variables
    #variables=all_variables
#)


# Rename columns immediately for clarity
#df = df.rename(columns={**total_by_race_vars, **median_home_value_by_race_vars, **ECON_VARS})

#print(df.head())

# In 2020 there were no 1-year estimates published due to Covid-19
#vintages = [year for year in range(2015, 2024) if year != 2020]
#df = download_multiyear(
    #dataset=ACS1,
    #vintages=vintages,
    #group="B02001",
    #state=NY,
    #county=NASSAU
#)
#print(df.head())

VINTAGE = 2024

# Race: https://api.census.gov/data/2024/acs/acs1/groups/B02001.html
GROUP = "B02001"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nRace")
print(df_al)

# Hispanic or Latino Origin: https://api.census.gov/data/2024/acs/acs1/groups/B03003.html
GROUP = "B03003"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nHispanic or Latino Origin")
print(df_al)

# Median Gross Rent by Bedrooms: https://api.census.gov/data/2024/acs/acs1/groups/B25031.html
GROUP = "B25031"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nMedian Gross Rent by Bedrooms")
print(df_al)

# Sex by Age: https://api.census.gov/data/2024/acs/acs1/groups/B01001.html
GROUP = "B01001"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nSex by Age")
print(df_al)

# Median Age by Sex: https://api.census.gov/data/2024/acs/acs1/groups/B01002.html
GROUP = "B01002"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nMedian Age by Sex")
print(df_al)

# Monthly Owner Costs (Dollars) by Mortgage Status: https://api.census.gov/data/2024/acs/acs1/groups/B25088.html
GROUP = "B25088"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nMonthly Owner Costs (Dollars)")
print(df_al)

# Geographical Mobility in the Past Year by Individual Income in the Past 12 Months: https://api.census.gov/data/2024/acs/acs1/groups/B07010.html
GROUP = "B07010"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nGeographical Mobility in the Past Year by Individual Income in the Past 12 Months")
print(df_al)

# Median Income in the Past 12 Months by Geographical Mobility in the Past Year: https://api.census.gov/data/2024/acs/acs1/groups/B07011.html
GROUP = "B07011"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nMedian Income in the Past 12 Months by Geographical Mobility in the Past Year")
print(df_al)

# Geographical Mobility in the Past Year by Poverty Status: https://api.census.gov/data/2024/acs/acs1/groups/B07012.html
GROUP = "B07012"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nGeographical Mobility in the Past Year by Poverty Status")
print(df_al)

# Means of Transportation to Work
GROUP = "B08006"
df_al = ced.download(ACS1, VINTAGE, group=GROUP, state=NY, county=QUEENS)
df_al = df_al.rename(columns=name_mapper)
print("\nMeans of Transportation to Work")
print(df_al)
