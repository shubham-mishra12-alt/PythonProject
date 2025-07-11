import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re # For regular expressions to find year columns

st.set_page_config(layout="wide", page_title="📊 Drone Delivery Analysis: Problem & Solution")

st.title("🚁 Drone Delivery Analysis: Solving the Pharmacy Travel Burden")
st.markdown("""
This application presents an analysis of the problem customers face with pharmacy travel and
how drone delivery offers a compelling solution, quantifying its impact on time and travel.
Use the filters in the sidebar to explore the data.
""")

# Load and limit data for performance
@st.cache_data
def load_data():
    df = pd.read_csv("Residential_Analysis.csv")
    # Limiting to a reasonable number of rows for faster prototyping/testing
    return df.iloc[:27161]

df = load_data()

# Drop unnamed columns that might result from CSV export issues
df = df.drop(columns=[col for col in df.columns if "Unnamed" in col], errors='ignore')

# --- Sidebar for Global Filters ---
st.sidebar.header("Global Filters")

# 1. Pharmacy Selection Filter (Multiselect)
all_pharmacies = df['Nearest Pharmacy'].unique().tolist()
selected_pharmacies = st.sidebar.multiselect(
    "1️⃣ Select Pharmacies:",
    options=all_pharmacies,
    default=all_pharmacies
)

# 2. Roundtrip Travel Distance Filter (Slider)
min_distance = df["Roundtrip Travel Distance to Nearest Pharmacy (miles)"].min()
max_distance = df["Roundtrip Travel Distance to Nearest Pharmacy (miles)"].max()
distance_range = st.sidebar.slider(
    "2️⃣ Roundtrip Travel Distance (miles):",
    min_value=float(min_distance),
    max_value=float(max_distance),
    value=(float(min_distance), float(max_distance))
)

# 3. Total Residents per Residence Filter (Slider)
min_residents = df['Total Residents/\nResidence'].min()
max_residents = df['Total Residents/\nResidence'].max()
total_residents_range = st.sidebar.slider(
    "3️⃣ Total Residents per Residence:",
    min_value=float(min_residents),
    max_value=float(max_residents),
    value=(float(min_residents), float(max_residents))
)

# 4. Serviceable Drone Market Status Filter (Radio Buttons)
serviceable_status_filter = st.sidebar.radio(
    "4️⃣ Drone Serviceability Status:",
    ('All Residences', 'Serviceable Only', 'Not Serviceable Only')
)

# --- Apply Global Filters to DataFrame ---
# Start with a copy to avoid modifying original cached DataFrame
df_filtered = df.copy()

# Apply Pharmacy filter
df_filtered = df_filtered[df_filtered['Nearest Pharmacy'].isin(selected_pharmacies)]

# Apply Distance filter
df_filtered = df_filtered[
    (df_filtered["Roundtrip Travel Distance to Nearest Pharmacy (miles)"] >= distance_range[0]) &
    (df_filtered["Roundtrip Travel Distance to Nearest Pharmacy (miles)"] <= distance_range[1])
]

# Apply Total Residents filter
df_filtered = df_filtered[
    (df_filtered['Total Residents/\nResidence'] >= total_residents_range[0]) &
    (df_filtered['Total Residents/\nResidence'] <= total_residents_range[1])
]

# --- Story Section 1: The Challenge ---
st.header("1️⃣ The Challenge: The Burden of Pharmacy Travel")
st.markdown("""
Customers today often face significant time and distance burdens when traveling to pharmacies.
Understanding this challenge is the first step towards a more efficient solution.
""")

# --- Plot 1: Histogram - Roundtrip Travel Distance to Nearest Pharmacy ---
st.subheader("1.1. Roundtrip Travel Distance to Nearest Pharmacy")
st.write("""
This histogram shows the distribution of roundtrip travel distances for residences to their nearest pharmacy.
It highlights how far people are currently traveling.
""")

if not df_filtered.empty:
    fig, ax = plt.subplots(figsize=(10, 6))

    counts, bins, patches = ax.hist(
        df_filtered["Roundtrip Travel Distance to Nearest Pharmacy (miles)"],
        bins=30,
        color="salmon",
        edgecolor="black"
    )

    ax.set_xlabel("Distance (miles)")
    ax.set_ylabel("Number of Residences")
    ax.set_title("Distribution of Roundtrip Travel Distance to Nearest Pharmacy")

    # Add data labels
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    for count, x in zip(counts, bin_centers):
        if count > 0:
            ax.text(x, count, int(count), ha='center', va='bottom', fontsize=8)

    st.pyplot(fig)

    avg_distance = df_filtered["Roundtrip Travel Distance to Nearest Pharmacy (miles)"].mean()
    st.markdown(f"*Average Roundtrip Distance (for filtered data):* {avg_distance:.2f} miles")
    st.markdown(f"*Total Residences considered for this plot:* {len(df_filtered)}")
else:
    st.info("No data to display Plot 1.1 for the selected global filters. Please adjust your selections.")


# --- Story Section 2: The Solution ---
st.header("2️⃣ The Solution: Introducing Drone Delivery")
st.markdown("""
Drone delivery offers a revolutionary way to address the challenges of pharmacy travel.
Let's explore its potential reach and how it can serve existing demand.
""")

# --- Plot 2: Pie Chart - The Reach of Drone Delivery ---
st.subheader("2.1. The Reach of Drone Delivery")
st.write("""
This pie chart illustrates the proportion of residences that fall within the serviceable drone market for a selected year.
""")

# Identify available years for Serviceable Drone Market
serviceable_drone_cols = [col for col in df.columns if re.match(r'Serviceable Drone Market, (\d{4})$', col)]
available_drone_years = sorted([int(re.search(r'(\d{4})$', col).group(1)) for col in serviceable_drone_cols])

selected_drone_market_year = None
if available_drone_years:
    selected_drone_market_year = st.selectbox(
        "Select Year for Serviceable Drone Market:",
        options=available_drone_years,
        # Default to 2027 if available, otherwise pick the last available year
        index=available_drone_years.index(2027) if 2027 in available_drone_years else (len(available_drone_years) -1 if available_drone_years else 0),
        key='drone_market_year_select'
    )
    drone_col_name = f'Serviceable Drone Market, {selected_drone_market_year}'
else:
    st.warning("No 'Serviceable Drone Market' columns with yearly data found.")
    drone_col_name = None

# For this pie chart, we apply global filters but not 'serviceable_status_filter' directly
# as the chart's purpose is to show the 'Yes'/'No' breakdown.
if not df_filtered.empty and drone_col_name and drone_col_name in df_filtered.columns:
    drone_counts = df_filtered[drone_col_name].value_counts().sort_index()

    if not drone_counts.empty:
        labels_map = {1: "Yes (Serviceable)", 0: "No (Not Serviceable)"}
        filtered_labels = []
        filtered_values = []
        if 1 in drone_counts.index:
            filtered_labels.append(labels_map[1])
            filtered_values.append(drone_counts[1])
        if 0 in drone_counts.index:
            filtered_labels.append(labels_map[0])
            filtered_values.append(drone_counts[0])

        if filtered_values:
            fig, ax = plt.subplots(figsize=(8, 8))
            wedges, texts, autotexts = ax.pie(filtered_values, labels=filtered_labels, autopct='%1.1f%%', startangle=90, pctdistance=0.85, colors=['lightskyblue', 'lightgray'])
            ax.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
            ax.set_title(f"Proportion of Serviceable Drone Market, {selected_drone_market_year}")

            plt.setp(autotexts, size=10, weight="bold", color="white")
            plt.setp(texts, size=10, color="black")

            st.pyplot(fig)
            total_residences_pie = drone_counts.sum()
            serviceable_count = drone_counts.get(1, 0)
            serviceable_percentage = (serviceable_count / total_residences_pie * 100) if total_residences_pie > 0 else 0
            st.markdown(f"*Total Residences considered for this plot:* {total_residences_pie}")
            st.markdown(f"*Percentage Serviceable (for selected filters):* {serviceable_percentage:.1f}%")
        else:
            st.info("No data to display Plot 2.1 for the selected year and filters.")
    else:
        st.info(f"No serviceable drone market data for {selected_drone_market_year} with current filters to display Plot 2.1.")
else:
    st.info("No data found to display Plot 2.1 after applying global filters or 'Serviceable Drone Market' column not found for selected year. Please adjust your selections.")


# --- Plot 3: Bar Chart - Number of Residences per Nearest Pharmacy ---
st.subheader("2.2. Demand per Pharmacy: Number of Residences Served")
st.write("""
This bar chart displays the number of residences associated with each nearest pharmacy,
giving an indication of the current demand handled by each location.
""")

if not df_filtered.empty:
    pharmacy_counts = df_filtered['Nearest Pharmacy'].value_counts().sort_index()

    if not pharmacy_counts.empty:
        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.bar(pharmacy_counts.index, pharmacy_counts.values, color='cornflowerblue')
        
        ax.set_xlabel("Nearest Pharmacy")
        ax.set_ylabel("Number of Residences")
        ax.set_title("Number of Residences per Nearest Pharmacy")
        plt.xticks(rotation=45, ha='right')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax.text(bar.get_x() + bar.get_width()/2, yval + 5, int(yval), ha='center', va='bottom', fontsize=8) # Adjust 5 for label spacing

        st.pyplot(fig)
        st.markdown(f"*Total Residences considered for this plot:* {len(df_filtered)}")
    else:
        st.info("No data for 'Nearest Pharmacy' found after applying global filters to display Plot 2.2. This might mean all residences were filtered out or have no nearest pharmacy data. Please adjust your selections.")
else:
    st.info("No data to display Plot 2.2 after applying global filters. This means all residences were filtered out. Please adjust your selections.")


# --- Story Section 3: Quantifying the Impact ---
st.header("3️⃣ Quantifying the Impact: Time and Travel Savings")
st.markdown("""
Drone delivery is not just convenient; it provides measurable benefits in terms of time saved
and reduction in vehicle travel, directly addressing the customer burden.
""")

# Identify available years for Annual Travel Time trends
annual_time_cols = [col for col in df.columns if re.match(r'Annual Travel Time to Pharmacy, (\d{4}) \(Minutes\)', col)]
available_annual_years = sorted([int(re.search(r'(\d{4})', col).group(1)) for col in annual_time_cols])

# Identify available years for Annual Vehicle Travel Replaced
annual_miles_replaced_cols = [col for col in df.columns if re.match(r'Annual Vehicle Travel Replaced by Drone Delivery, (\d{4}) \(Miles\)', col)]
available_miles_replaced_years = sorted([int(re.search(r'(\d{4})', col).group(1)) for col in annual_miles_replaced_cols])


# --- Plot 4: GROUPED BAR CHART: Annual Travel Time Trends (Pharmacy vs. Drone Replaced) ---
st.subheader("3.1. Annual Time Savings with Drone Delivery")
st.write("""
This grouped bar chart displays the average annual travel time to pharmacies and the average annual travel time replaced by drone delivery for selected years.
It clearly highlights the potential efficiency gains and time savings for customers over time.
""")

selected_years_for_trends = st.multiselect(
    "Select Years for Annual Travel Time Trends:",
    options=available_annual_years,
    default=available_annual_years,
    key='years_for_travel_trends'
)

if selected_years_for_trends:
    avg_pharmacy_travel_times = []
    avg_drone_replaced_travel_times = []
    years_to_plot = sorted(selected_years_for_trends)
    
    # Apply serviceable_status_filter to a copy of df_filtered for this plot
    df_plot_data_time_trends = df_filtered.copy()
    if serviceable_status_filter == 'Serviceable Only':
        # Apply serviceable filter based on the first selected year for consistency across years for this plot
        first_year_col = f'Serviceable Drone Market, {years_to_plot[0]}' if years_to_plot else None
        if first_year_col and first_year_col in df_plot_data_time_trends.columns:
            df_plot_data_time_trends = df_plot_data_time_trends[df_plot_data_time_trends[first_year_col] == 1]
        else:
            st.warning(f"Serviceable Drone Market data for {years_to_plot[0]} not found. Filter might not be applied correctly, leading to no data for this plot.")
            df_plot_data_time_trends = pd.DataFrame() # Clear data if column missing
    elif serviceable_status_filter == 'Not Serviceable Only':
        first_year_col = f'Serviceable Drone Market, {years_to_plot[0]}' if years_to_plot else None
        if first_year_col and first_year_col in df_plot_data_time_trends.columns:
            df_plot_data_time_trends = df_plot_data_time_trends[df_plot_data_time_trends[first_year_col] == 0]
        else:
            st.warning(f"Serviceable Drone Market data for {years_to_plot[0]} not found. Filter might not be applied correctly, leading to no data for this plot.")
            df_plot_data_time_trends = pd.DataFrame() # Clear data if column missing


    if not df_plot_data_time_trends.empty:
        for year in years_to_plot:
            pharmacy_col = f'Annual Travel Time to Pharmacy, {year} (Minutes)'
            drone_col = f'Annual Travel Time Replaced by Drone Delivery, {year} (Minutes)'
            
            if pharmacy_col in df_plot_data_time_trends.columns and drone_col in df_plot_data_time_trends.columns:
                avg_pharmacy_travel_times.append(df_plot_data_time_trends[pharmacy_col].mean())
                avg_drone_replaced_travel_times.append(df_plot_data_time_trends[drone_col].mean())
            else:
                # Append 0 if column missing to avoid errors, plot will show missing data
                avg_pharmacy_travel_times.append(0)
                avg_drone_replaced_travel_times.append(0)

        if any(p > 0 for p in avg_pharmacy_travel_times) or any(d > 0 for d in avg_drone_replaced_travel_times):
            fig, ax = plt.subplots(figsize=(12, 7))
            bar_width = 0.35
            index = np.arange(len(years_to_plot))

            bar1 = ax.bar(index - bar_width/2, avg_pharmacy_travel_times, bar_width, label='Avg. Annual Travel Time to Pharmacy', color='mediumseagreen')
            bar2 = ax.bar(index + bar_width/2, avg_drone_replaced_travel_times, bar_width, label='Avg. Annual Time Replaced by Drone', color='steelblue')

            ax.set_xlabel("Year")
            ax.set_ylabel("Average Time (Minutes)")
            ax.set_title("Average Annual Travel Time: Pharmacy vs. Drone Replaced")
            ax.set_xticks(index)
            ax.set_xticklabels(years_to_plot)
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            for bars_group in [bar1, bar2]:
                for bar_item in bars_group:
                    yval = bar_item.get_height()
                    if yval > 0.01:
                        ax.text(bar_item.get_x() + bar_item.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

            st.pyplot(fig)
            
            if years_to_plot:
                first_year = years_to_plot[0]
                pharm_col = f'Annual Travel Time to Pharmacy, {first_year} (Minutes)'
                drone_rep_col = f'Annual Travel Time Replaced by Drone Delivery, {first_year} (Minutes)'
                
                if pharm_col in df_plot_data_time_trends.columns and drone_rep_col in df_plot_data_time_trends.columns:
                    total_time_saved_series = df_plot_data_time_trends[pharm_col] - df_plot_data_time_trends[drone_rep_col]
                    avg_time_saved_first_year = total_time_saved_series.mean()
                    st.markdown(f"*Average Annual Time Saved per Residence (in {first_year} for filtered data):* {avg_time_saved_first_year:.1f} minutes")
                st.markdown(f"*Total Residences considered for this plot:* {len(df_plot_data_time_trends)}")

        else:
            st.info("No data available for the selected years to display Plot 3.1 Annual Travel Time Trends with current filters. This might be due to filter settings or missing data columns.")
    else:
        st.info("No data found after applying filters for Plot 3.1 'Annual Time Savings with Drone Delivery'. Please adjust filters.")
else:
    st.info("Please select at least one year to display Plot 3.1 Annual Travel Time Trends.")


# --- Plot 5: BAR CHART: Annual Vehicle Miles Saved by Drones ---
st.subheader("3.2. Annual Vehicle Miles Saved by Drones")
st.write("""
This bar chart displays the average annual vehicle travel (in miles) that can be replaced by drone delivery for the selected years.
This represents a tangible benefit in terms of reduced fuel consumption, lower vehicle wear-and-tear, and environmental impact.
""")

selected_years_for_miles_saved = st.multiselect(
    "Select Years for Annual Vehicle Miles Saved:",
    options=available_miles_replaced_years,
    default=available_miles_replaced_years,
    key='years_for_miles_saved'
)

if selected_years_for_miles_saved:
    avg_miles_replaced = []
    years_to_plot_miles = sorted(selected_years_for_miles_saved)

    # Apply serviceable_status_filter to a copy of df_filtered for this plot
    df_plot_data_miles_saved = df_filtered.copy()
    if serviceable_status_filter == 'Serviceable Only':
        first_year_col_miles = f'Serviceable Drone Market, {years_to_plot_miles[0]}' if years_to_plot_miles else None
        if first_year_col_miles and first_year_col_miles in df_plot_data_miles_saved.columns:
            df_plot_data_miles_saved = df_plot_data_miles_saved[df_plot_data_miles_saved[first_year_col_miles] == 1]
        else:
            st.warning(f"Serviceable Drone Market data for {years_to_plot_miles[0]} not found. Filter might not be applied correctly, leading to no data for this plot.")
            df_plot_data_miles_saved = pd.DataFrame() # Clear data if column missing
    elif serviceable_status_filter == 'Not Serviceable Only':
        first_year_col_miles = f'Serviceable Drone Market, {years_to_plot_miles[0]}' if years_to_plot_miles else None
        if first_year_col_miles and first_year_col_miles in df_plot_data_miles_saved.columns:
            df_plot_data_miles_saved = df_plot_data_miles_saved[df_plot_data_miles_saved[first_year_col_miles] == 0]
        else:
            st.warning(f"Serviceable Drone Market data for {years_to_plot_miles[0]} not found. Filter might not be applied correctly, leading to no data for this plot.")
            df_plot_data_miles_saved = pd.DataFrame() # Clear data if column missing


    if not df_plot_data_miles_saved.empty:
        for year in years_to_plot_miles:
            miles_col = f'Annual Vehicle Travel Replaced by Drone Delivery, {year} (Miles)'
            if miles_col in df_plot_data_miles_saved.columns:
                avg_miles_replaced.append(df_plot_data_miles_saved[miles_col].mean())
            else:
                avg_miles_replaced.append(0) # Append 0 if column missing

        if any(m > 0 for m in avg_miles_replaced):
            fig, ax = plt.subplots(figsize=(12, 7))
            bars = ax.bar(years_to_plot_miles, avg_miles_replaced, color='darkcyan')

            ax.set_xlabel("Year")
            ax.set_ylabel("Average Miles Saved")
            ax.set_title("Average Annual Vehicle Miles Saved by Drones")
            ax.set_xticks(years_to_plot_miles)
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            for bar_item in bars:
                yval = bar_item.get_height()
                if yval > 0.01:
                    ax.text(bar_item.get_x() + bar_item.get_width()/2, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom', fontsize=8)

            st.pyplot(fig)
            st.markdown(f"*Total Residences considered for this plot:* {len(df_plot_data_miles_saved)}")
        else:
            st.info("No data available for the selected years to display Plot 3.2 Annual Vehicle Miles Saved with current filters. This might be due to filter settings or missing data columns.")
    else:
        st.info("No data found after applying filters for Plot 3.2 'Annual Vehicle Miles Saved by Drones'. Please adjust filters.")
else:
    st.info("Please select at least one year to display Plot 3.2 Annual Vehicle Miles Saved chart.")


st.markdown("""
---
*Note on Data & Interactivity:*
All plots are dynamically filtered by the global selections in the sidebar.
Year-specific data (for 'Serviceable Drone Market', 'Annual Travel Time', and 'Annual Vehicle Travel Replaced')
is pulled from the dataset based on the selected year(s) for each respective chart.
""")
