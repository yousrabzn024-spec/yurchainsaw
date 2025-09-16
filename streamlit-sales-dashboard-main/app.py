import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------
# Streamlit page configuration
# ----------------------------
st.set_page_config(page_title="Stylkom Dashboard", page_icon=":sparkles:", layout="wide")

# ----------------------------
# Custom CSS and logos
# ----------------------------
st.markdown("""
    <style>
    body {
        font-family: 'Space Grotesk', sans-serif;
        background-color: #E3D2FF;
    }
    .sidebar-logo {
        text-align: center;
        margin-top: -30px;
        margin-bottom: 20px;
    }
    .main-logo {
        text-align: center;
        margin-top: -50px;
        margin-bottom: 30px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-size: 18px;
        text-align: left;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
    }
    th {
        background-color: #f2f2f2;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar logo and main logo (optional - will error if file not found)
try:
    st.sidebar.image("logo.png", width=250)
    st.image("logo.png", width=350)
except Exception:
    # ignore if logo not present
    pass

# ----------------------------
# Data loader with normalization
# ----------------------------
@st.cache_data
def load_data(file):
    # accept both csv and xlsx
    if hasattr(file, "name") and file.name.lower().endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    # normalize column names to uppercase and strip spaces
    df.columns = df.columns.str.strip().str.upper()

    # ensure date/time formatting
    if "BILL_DATE" in df.columns:
        df["BILL_DATE"] = pd.to_datetime(df["BILL_DATE"], errors='coerce')

    if "BILL_TIME" in df.columns:
        # try flexible parsing for different time formats
        df["BILL_TIME"] = df["BILL_TIME"].astype(str)
        df["HOUR"] = pd.to_datetime(df["BILL_TIME"], errors='coerce').dt.hour
    else:
        df["HOUR"] = pd.NA

    # ensure numeric columns are numeric (fill NaN with 0 where appropriate)
    numeric_cols = ["NET_TOTAL", "QTY", "RETAIL_PRICE", "DIS_AMT", "AMOUNT", "VAT_AMT"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    # ensure text columns exist to avoid KeyErrors later
    for c in ["MODEL_CODE", "COLOR_NAME", "SEASON", "PAYMENTTYPE", "TERMINAL", "CATEGORY"]:
        if c not in df.columns:
            df[c] = pd.NA

    return df

# ----------------------------
# Helper: safe get unique values
# ----------------------------
def uniq_or_empty(df, col):
    return df[col].unique() if col in df.columns else []

# ----------------------------
# Main
# ----------------------------
uploaded_file = st.sidebar.file_uploader("Upload your sales data", type=["csv", "xlsx"]) 

if uploaded_file is not None:
    df = load_data(uploaded_file)

    # Sidebar filters
    st.sidebar.header("Please Filter Here:")

    month_opts = uniq_or_empty(df, "BILL_DATE")
    if len(month_opts) > 0 and "BILL_DATE" in df.columns:
        month_list = df["BILL_DATE"].dt.month_name().unique()
    else:
        month_list = []

    month = st.sidebar.multiselect(
        "Select the Month:",
        options=month_list,
        default=month_list
    )

    day_list = df["BILL_DATE"].dt.day.unique() if "BILL_DATE" in df.columns else []
    day = st.sidebar.multiselect(
        "Select the Day:",
        options=day_list,
        default=day_list
    )

    brand_list = uniq_or_empty(df, "BRAND")
    brand = st.sidebar.multiselect(
        "Select the Brand:",
        options=brand_list,
        default=brand_list
    )

    store_list = uniq_or_empty(df, "STORE_NAME")
    store_name = st.sidebar.multiselect(
        "Select the Store:",
        options=store_list,
        default=store_list
    )

    # Build filter mask safely
    mask = pd.Series(True, index=df.index)
    if "BILL_DATE" in df.columns and month:
        mask &= df["BILL_DATE"].dt.month_name().isin(month)
    if "BILL_DATE" in df.columns and list(day):
        mask &= df["BILL_DATE"].dt.day.isin(day)
    if "BRAND" in df.columns and list(brand):
        mask &= df["BRAND"].isin(brand)
    if "STORE_NAME" in df.columns and list(store_name):
        mask &= df["STORE_NAME"].isin(store_name)

    df_selection = df[mask].copy()

    # Check if the filtered data is empty
    if df_selection.empty:
        st.warning("No data available based on the current filter settings!")
        st.stop()

    # Main Page Title
    st.title("Sales Dashboard")
    st.markdown("##")

    # Store sqm mapping (uppercase keys)
    store_sqm = {
        "Showroom Koton": 950, "SUWEN": 130, "STYLKOM": 650, "SETRE": 250,
        "PANCO": 150, "MAVI": 250, "CITY STYLE": 250,
        "DS DAMAT": 140, "KIGILI": 200, "NINE WEST": 120
    }

    # ----------------------------
    # Calculate KPIs
    # ----------------------------
    total_sales = float(df_selection["NET_TOTAL"].sum()) if "NET_TOTAL" in df_selection.columns else 0.0
    total_items_sold = int(df_selection["QTY"].sum()) if "QTY" in df_selection.columns else 0

    try:
        best_selling_model_code = df_selection.groupby("MODEL_CODE")["QTY"].sum().idxmax() if ("MODEL_CODE" in df_selection.columns and "QTY" in df_selection.columns and not df_selection.groupby("MODEL_CODE")["QTY"].sum().empty) else "N/A"
    except Exception:
        best_selling_model_code = "N/A"

    if ("DIS_AMT" in df_selection.columns) and ("RETAIL_PRICE" in df_selection.columns) and df_selection["RETAIL_PRICE"].sum() != 0:
        average_discount = round((df_selection[df_selection["DIS_AMT"] != 0]["DIS_AMT"].sum() / df_selection[df_selection["DIS_AMT"] != 0]["RETAIL_PRICE"].sum()) * 100, 2)
    else:
        average_discount = 0.0

    num_bills = int(df_selection["BILL_NO"].nunique()) if "BILL_NO" in df_selection.columns else 0
    atv = total_sales / num_bills if num_bills > 0 else 0
    upt = total_items_sold / num_bills if num_bills > 0 else 0
    asp = df_selection["AMOUNT"].sum() / total_items_sold if ("AMOUNT" in df_selection.columns and total_items_sold > 0) else 0

    total_units = df_selection[df_selection["QTY"] > 0]["QTY"].sum() if "QTY" in df_selection.columns else 0
    returns = abs(df_selection[df_selection["QTY"] < 0]["QTY"].sum()) if "QTY" in df_selection.columns else 0
    return_rate = (returns / total_units) * 100 if total_units > 0 else 0

    # Payment type percentages (card vs cash)
    card_sales = 0.0
    cash_sales = 0.0
    if "PAYMENTTYPE" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        # flexible matching for card vs cash
        card_mask = df_selection["PAYMENTTYPE"].astype(str).str.contains("CARD|CB|VISA|MASTERCARD|POS", case=False, na=False)
        cash_mask = df_selection["PAYMENTTYPE"].astype(str).str.contains("CASH|ESPECE|ESP", case=False, na=False)
        card_sales = float(df_selection.loc[card_mask, "NET_TOTAL"].sum())
        cash_sales = float(df_selection.loc[cash_mask, "NET_TOTAL"].sum())
        # If neither matched, attempt simple split by unique values
        if card_sales == 0 and cash_sales == 0:
            # fallback: choose highest two types as card/cash guesses
            group = df_selection.groupby("PAYMENTTYPE")["NET_TOTAL"].sum().reset_index().sort_values("NET_TOTAL", ascending=False)
            if not group.empty:
                # treat top as 'card' and second as 'cash' as fallback (best-effort)
                card_sales = float(group.iloc[0]["NET_TOTAL"])
                if len(group) > 1:
                    cash_sales = float(group.iloc[1]["NET_TOTAL"])
    pct_card = (card_sales / total_sales) * 100 if total_sales > 0 else 0
    pct_cash = (cash_sales / total_sales) * 100 if total_sales > 0 else 0

    # Sales per sqm (align store names uppercase)
    sales_per_sqm = None
    if "STORE_NAME" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        sales_by_store = df_selection.groupby(df_selection["STORE_NAME"].str.upper())["NET_TOTAL"].sum()
        
        sqm_series = pd.Series(store_sqm)
        sqm_series.index = sqm_series.index.str.upper()

        # convert sqm_series to float series with index uppercase
        sqm_series = pd.Series(store_sqm)
        sqm_series.index = sqm_series.index.str.upper()
        common_index = sales_by_store.index.intersection(sqm_series.index)
        if not common_index.empty:
            sales_per_sqm = (sales_by_store.loc[common_index] / sqm_series.loc[common_index])
        else:
            sales_per_sqm = pd.Series([], dtype=float)
    else:
        sales_per_sqm = pd.Series([], dtype=float)

    # Display KPIs in a nice table (transparent background) - added payment % columns
    st.markdown("### Key Performance Indicators")
    st.markdown(f"""
     <table style="background-color: transparent; width: 100%; border-collapse: collapse; font-size: 18px; text-align: left;">
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Total Sales</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">DA {total_sales:,.0f}</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Total Items Sold</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{total_items_sold:,}</td>
        </tr>
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Best Selling Model</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{best_selling_model_code}</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Average Discount %</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{average_discount:.2f}%</td>
        </tr>
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Number of Bills</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{num_bills:,}</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">ATV</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">DA {atv:.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">UPT</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{upt:.2f}</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">ASP</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">DA {asp:.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Return %</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{return_rate:.2f}%</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">Sales per Sqm (avg)</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">DA {sales_per_sqm.mean():.2f}</td>
        </tr>
        <tr style="border: 1px solid #ddd; background-color: transparent;">
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">% Card Sales</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{pct_card:.2f}%</td>
            <th style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">% Cash Sales</th>
            <td style="padding: 8px; border: 1px solid #ddd; background-color: transparent;">{pct_cash:.2f}%</td>
        </tr>
    </table>
    """, unsafe_allow_html=True)

    # ----------------------------
    # Sales by Gender (ATV)
    # ----------------------------
    if "GENDER" in df_selection.columns:
        gender_sales = df_selection.groupby("GENDER").agg(
            total_sales=("NET_TOTAL", "sum"),
            num_bills=("BILL_NO", "nunique")
        ).reset_index()
        gender_sales["ATV"] = gender_sales.apply(lambda r: r["total_sales"] / r["num_bills"] if r["num_bills"]>0 else 0, axis=1)

        st.markdown("### Sales by Gender")
        st.dataframe(gender_sales[["GENDER", "total_sales", "ATV"]])

    # ----------------------------
    # Sell-Out % Calculation
    # ----------------------------
    st.markdown("### Sell Out % Calculation")
    col_store, col_stock = st.columns(2)
    with col_store:
        selected_store = st.selectbox("Select Store:", options=df["STORE_NAME"].unique() if "STORE_NAME" in df.columns else ["N/A"]) 
    with col_stock:
        stock_quantity = st.number_input("Enter Stock Quantity:", min_value=0, value=0)

    total_sold = df_selection[df_selection["STORE_NAME"] == selected_store]["QTY"].sum() if ("STORE_NAME" in df_selection.columns and "QTY" in df_selection.columns) else 0
    sell_out_percentage = (total_sold / stock_quantity) * 100 if stock_quantity > 0 else 0
    st.markdown(f"**Sell Out % for {selected_store}: {sell_out_percentage:.2f}%**")

    if stock_quantity > 0 and total_sold > 0 and "GENDER" in df_selection.columns:
        sell_out_data = df_selection[df_selection["STORE_NAME"] == selected_store].groupby("GENDER")["QTY"].sum().reset_index()
        sell_out_data["Sell Out %"] = (sell_out_data["QTY"] / stock_quantity) * 100

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sell_out_data['GENDER'],
            y=sell_out_data['Sell Out %'],
            text=sell_out_data['Sell Out %'].round(2),
            textposition='auto'
        ))
        fig.update_layout(
            title_text=f"Sell Out % by Gender for {selected_store}",
            xaxis_title='Gender',
            yaxis_title='Sell Out %',
            yaxis=dict(range=[0, 100]),
            template='plotly_white'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        if stock_quantity == 0:
            st.warning("Please enter a stock quantity greater than zero to calculate Sell Out %.")

    # ----------------------------
    # Types By Gender (HR1)
    # ----------------------------
    if "HR1" in df_selection.columns and "GENDER" in df_selection.columns:
        st.markdown("### Types By Gender")
        selected_gender = st.selectbox("Select Gender:", options=df_selection["GENDER"].unique())
        gender_data = df_selection[df_selection["GENDER"] == selected_gender]
        hr1_sales = gender_data.groupby("HR1")["NET_TOTAL"].sum().reset_index()
        if not hr1_sales.empty:
            fig_hr1_pie = px.pie(hr1_sales, values="NET_TOTAL", names="HR1", title=f"Sales Distribution by Type (HR1) for {selected_gender}")
            st.plotly_chart(fig_hr1_pie, use_container_width=True)
        else:
            st.warning(f"No sales data available for {selected_gender}.")

    # ----------------------------
    # Sales by Hour
    # ----------------------------
    if "HOUR" in df_selection.columns:
        sales_by_hour = df_selection.groupby("HOUR")["NET_TOTAL"].sum().reset_index()
        if not sales_by_hour.empty:
            max_sales_hour = sales_by_hour["NET_TOTAL"].max()
            sales_by_hour["color"] = sales_by_hour["NET_TOTAL"].apply(lambda x: "Green" if x == max_sales_hour else "Blue")
            fig_hourly_sales = px.bar(
                sales_by_hour,
                x="HOUR",
                y="NET_TOTAL",
                title="Sales by Hour",
                color="color",
                color_discrete_map={"Blue": "#0083b8", "Green": "#d7f277"},
                template="plotly_white",
            )
            fig_hourly_sales.update_layout(xaxis=dict(tickmode="linear"), plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(showgrid=False))
            st.plotly_chart(fig_hourly_sales, use_container_width=True)

    # ----------------------------
    # Sales by HR1
    # ----------------------------
    if "HR1" in df_selection.columns:
        sales_by_hr1 = df_selection.groupby("HR1")["NET_TOTAL"].sum().reset_index()
        if not sales_by_hr1.empty:
            max_sales_hr1 = sales_by_hr1["NET_TOTAL"].max()
            sales_by_hr1["color"] = sales_by_hr1["NET_TOTAL"].apply(lambda x: "Green" if x == max_sales_hr1 else "Blue")
            fig_sales_hr1 = px.bar(
                sales_by_hr1,
                x="HR1",
                y="NET_TOTAL",
                title="Sales by Product Type (HR1)",
                color="color",
                color_discrete_map={"Blue": "#0083b8", "Green": "#d7f277"},
                template="plotly_white",
            )
            fig_sales_hr1.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False))
            st.plotly_chart(fig_sales_hr1, use_container_width=True)

    # ----------------------------
    # Returns & Sales over time
    # ----------------------------
    if "NET_TOTAL" in df_selection.columns:
        df_selection['RETURNED_AMOUNT'] = -df_selection['NET_TOTAL'].apply(lambda x: x if x < 0 else 0)
        sales_over_time = df_selection.groupby('BILL_DATE').agg(
            net_total=('NET_TOTAL', 'sum'),
            returned_amount=('RETURNED_AMOUNT', 'sum')
        ).reset_index()

        st.subheader('Line Plot: Sales Over Time')
        # remove rows with NaT bill_date
        sales_over_time = sales_over_time.dropna(subset=['BILL_DATE'])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sales_over_time['BILL_DATE'], y=sales_over_time['net_total'], mode='lines', name='Net Total', line=dict(color='#0083b8', width=2)))
        fig.add_trace(go.Scatter(x=sales_over_time['BILL_DATE'], y=sales_over_time['returned_amount'], mode='lines', name='Returned Amount', line=dict(color='red', width=2, dash='dot')))
        fig.update_layout(title='Sales Over Time', xaxis_title='Date', yaxis_title='Amount', legend_title='Metrics', template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # Sales by Store (colored)
    # ----------------------------
    if "STORE_NAME" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        sales_by_store = df_selection.groupby("STORE_NAME")["NET_TOTAL"].sum().reset_index()
        max_sales_store = sales_by_store["NET_TOTAL"].max()
        sales_by_store["color"] = sales_by_store["NET_TOTAL"].apply(lambda x: "Green" if x == max_sales_store else "Blue")
        fig_sales_store = px.bar(sales_by_store, x="STORE_NAME", y="NET_TOTAL", title="Sales by Store", color="color", color_discrete_map={"Blue": "#0083b8", "Green": "#d7f277"}, template="plotly_white")
        fig_sales_store.update_layout(plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(showgrid=False))

        left_column, right_column = st.columns(2)
        # left: hourly (if exists), right: sales by hr1 (if exists)
        if "HOUR" in df_selection.columns and 'fig_hourly_sales' in locals():
            left_column.plotly_chart(fig_hourly_sales, use_container_width=True)
        if 'fig_sales_hr1' in locals():
            right_column.plotly_chart(fig_sales_hr1, use_container_width=True)

        st.plotly_chart(fig_sales_store, use_container_width=True)

    # ----------------------------
    # Salesmen Performance using UPT
    # ----------------------------
    st.markdown("### Top 5 Salesmen by Store (Based on UPT)")
    if all(c in df_selection.columns for c in ["SALESMAN", "STORE_NAME", "QTY", "BILL_NO"]):
        salesmen_performance = df_selection.groupby(["STORE_NAME", "SALESMAN"]).agg(
            total_units=("QTY", "sum"),
            unique_bills=("BILL_NO", "nunique")
        ).reset_index()
        salesmen_performance["UPT"] = salesmen_performance.apply(lambda r: r["total_units"] / r["unique_bills"] if r["unique_bills"]>0 else 0, axis=1)
        salesmen_performance["TOTAL_STORE_UPT"] = salesmen_performance.groupby("STORE_NAME")["UPT"].transform("sum")
        salesmen_performance["PERCENTAGE"] = salesmen_performance.apply(lambda r: (r["UPT"] / r["TOTAL_STORE_UPT"] * 100) if r["TOTAL_STORE_UPT"]>0 else 0, axis=1)

        top_salesmen = salesmen_performance.sort_values(["STORE_NAME", "PERCENTAGE"], ascending=[True, False])
        top_salesmen = top_salesmen.groupby("STORE_NAME").head(5).reset_index(drop=True)

        selected_store_for_salesmen = st.selectbox("Select a Store to view Top 5 Salesmen:", top_salesmen["STORE_NAME"].unique())
        filtered_salesmen = top_salesmen[top_salesmen["STORE_NAME"] == selected_store_for_salesmen]

        st.write(f"Top 5 Salesmen for {selected_store_for_salesmen} (Based on UPT)")
        st.dataframe(filtered_salesmen[["SALESMAN", "UPT", "PERCENTAGE"]])

        fig_salesmen_pie = px.pie(filtered_salesmen, values="PERCENTAGE", names="SALESMAN", title=f"Top 5 Salesmen Performance in {selected_store_for_salesmen} (Based on UPT)")
        st.plotly_chart(fig_salesmen_pie, use_container_width=True)
    else:
        st.warning("The required columns for salesmen performance analysis are missing.")

    # ----------------------------
    # Sale Comparison (two weekly views)
    # ----------------------------
    st.markdown("<h3 style='text-align: center;'>Sale Comparison</h3>", unsafe_allow_html=True)
    st.markdown("Sales Plot 1: Select Filters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        year1 = st.selectbox("Year", options=df["BILL_DATE"].dt.year.unique() if "BILL_DATE" in df.columns else [], key="year1")
    with col2:
        month1 = st.selectbox("Month", options=df["BILL_DATE"].dt.month_name().unique() if "BILL_DATE" in df.columns else [], key="month1")

    # weeks of that month
    if "BILL_DATE" in df.columns and month1:
        month_index1 = pd.to_datetime(month1, format='%B').month
        month_start = df[(df["BILL_DATE"].dt.year == int(year1)) & (df["BILL_DATE"].dt.month == month_index1)]
        weeks_in_month1 = month_start['BILL_DATE'].dt.isocalendar().week.unique()
    else:
        weeks_in_month1 = []

    with col3:
        week1 = st.selectbox("Week", options=weeks_in_month1, key="week1")
    with col4:
        store1 = st.selectbox("Store", options=df["STORE_NAME"].unique() if "STORE_NAME" in df.columns else [], key="store1")

    # compute start and end date
    if "BILL_DATE" in df.columns and len(weeks_in_month1)>0:
        start_date1 = df[(df["BILL_DATE"].dt.year == int(year1)) & (df["BILL_DATE"].dt.isocalendar().week == int(week1))]["BILL_DATE"].min()
        end_date1 = start_date1 + pd.DateOffset(days=6)
    else:
        start_date1 = None
        end_date1 = None

    filtered_data1 = df.copy()
    if start_date1 is not None and end_date1 is not None and "STORE_NAME" in df.columns:
        filtered_data1 = df[(df["BILL_DATE"] >= start_date1) & (df["BILL_DATE"] <= end_date1) & (df["STORE_NAME"] == store1)]

    sales_by_day1 = pd.DataFrame()
    if not filtered_data1.empty and "BILL_DATE" in filtered_data1.columns:
        sales_by_day1 = filtered_data1.groupby(filtered_data1["BILL_DATE"].dt.date)["NET_TOTAL"].sum().reset_index()
        sales_by_day1['Day'] = pd.to_datetime(sales_by_day1['BILL_DATE']).dt.day_name()
        sales_by_day1['Day_BillDate'] = sales_by_day1['Day'] + ' (' + sales_by_day1['BILL_DATE'].astype(str) + ')'

    st.markdown("Sales Plot 2: Select Filters")
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        year2 = st.selectbox("Year", options=df["BILL_DATE"].dt.year.unique() if "BILL_DATE" in df.columns else [], key="year2")
    with col6:
        month2 = st.selectbox("Month", options=df["BILL_DATE"].dt.month_name().unique() if "BILL_DATE" in df.columns else [], key="month2")

    if "BILL_DATE" in df.columns and month2:
        month_index2 = pd.to_datetime(month2, format='%B').month
        month_start2 = df[(df["BILL_DATE"].dt.year == int(year2)) & (df["BILL_DATE"].dt.month == month_index2)]
        weeks_in_month2 = month_start2['BILL_DATE'].dt.isocalendar().week.unique()
    else:
        weeks_in_month2 = []

    with col7:
        week2 = st.selectbox("Week", options=weeks_in_month2, key="week2")
    with col8:
        store2 = st.selectbox("Store", options=df["STORE_NAME"].unique() if "STORE_NAME" in df.columns else [], key="store2")

    if "BILL_DATE" in df.columns and len(weeks_in_month2)>0:
        start_date2 = df[(df["BILL_DATE"].dt.year == int(year2)) & (df["BILL_DATE"].dt.isocalendar().week == int(week2))]["BILL_DATE"].min()
        end_date2 = start_date2 + pd.DateOffset(days=6)
    else:
        start_date2 = None
        end_date2 = None

    filtered_data2 = df.copy()
    if start_date2 is not None and end_date2 is not None and "STORE_NAME" in df.columns:
        filtered_data2 = df[(df["BILL_DATE"] >= start_date2) & (df["BILL_DATE"] <= end_date2) & (df["STORE_NAME"] == store2)]

    sales_by_day2 = pd.DataFrame()
    if not filtered_data2.empty and "BILL_DATE" in filtered_data2.columns:
        sales_by_day2 = filtered_data2.groupby(filtered_data2["BILL_DATE"].dt.date)["NET_TOTAL"].sum().reset_index()
        sales_by_day2['Day'] = pd.to_datetime(sales_by_day2['BILL_DATE']).dt.day_name()
        sales_by_day2['Day_BillDate'] = sales_by_day2['Day'] + ' (' + sales_by_day2['BILL_DATE'].astype(str) + ')'

    left_column, right_column = st.columns(2)
    with left_column:
        if not sales_by_day1.empty:
            fig_sales1 = px.bar(sales_by_day1, x="Day_BillDate", y="NET_TOTAL", title=f"Total Sales by Day for {month1} {year1} (Week {week1})", color="NET_TOTAL", color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig_sales1, use_container_width=True)
        else:
            st.info("No data for Plot 1 selection.")

    with right_column:
        if not sales_by_day2.empty:
            fig_sales2 = px.bar(sales_by_day2, x="Day_BillDate", y="NET_TOTAL", title=f"Total Sales by Day for {month2} {year2} (Week {week2})", color="NET_TOTAL", color_continuous_scale=px.colors.sequential.Viridis)
            st.plotly_chart(fig_sales2, use_container_width=True)
        else:
            st.info("No data for Plot 2 selection.")

    # ----------------------------
    # Best & Worst Selling Model (filtered by HR1)
    # ----------------------------
    st.markdown("<h3 style='text-align: center;'> Best & Worst Selling Model </h3>", unsafe_allow_html=True)
    if "HR1" in df.columns and "MODEL_CODE" in df.columns and "QTY" in df.columns:
        selected_hr1 = st.selectbox("Select HR1 Type:", options=df["HR1"].unique())
        filtered_hr1_data = df_selection[df_selection["HR1"] == selected_hr1]
        if not filtered_hr1_data.empty:
            model_sales = filtered_hr1_data.groupby("MODEL_CODE")["QTY"].sum().reset_index()
            top_models = model_sales.nlargest(10, "QTY")
            bottom_models = model_sales.nsmallest(10, "QTY")

            fig_top = px.pie(top_models, values='QTY', names='MODEL_CODE', title='Top 10 Best Selling Models', hole=0.4)
            fig_bottom = px.pie(bottom_models, values='QTY', names='MODEL_CODE', title='Top 10 Worst Selling Models', hole=0.4)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("### Top 10 Best Selling Models Table")
                st.dataframe(top_models)
            with col4:
                st.markdown("### Top 10 Worst Selling Models Table")
                st.dataframe(bottom_models)

            col1b, col2b = st.columns(2)
            with col1b:
                st.plotly_chart(fig_top, use_container_width=True)
            with col2b:
                st.plotly_chart(fig_bottom, use_container_width=True)
        else:
            st.info("No data for the selected HR1.")
    else:
        st.info("Model analysis requires HR1, MODEL_CODE and QTY columns.")

    # ----------------------------
    # NEW: % of sold colors for selected MODEL_CODE
    # ----------------------------
    st.markdown("### Color Distribution by Model")
    if "MODEL_CODE" in df_selection.columns and "COLOR_NAME" in df_selection.columns and ("QTY" in df_selection.columns or "NET_TOTAL" in df_selection.columns):
        model_list = df_selection["MODEL_CODE"].dropna().unique()
        selected_model = st.selectbox("Select Model Code:", options=model_list)
        if selected_model:
            # use QTY when available, otherwise NET_TOTAL
            metric_col = "QTY" if "QTY" in df_selection.columns and df_selection["QTY"].sum() > 0 else "NET_TOTAL"
            color_dist = df_selection[df_selection["MODEL_CODE"] == selected_model].groupby("COLOR_NAME")[metric_col].sum().reset_index().sort_values(metric_col, ascending=False)
            if not color_dist.empty:
                color_dist["pct"] = (color_dist[metric_col] / color_dist[metric_col].sum()) * 100
                fig_color = px.pie(color_dist, names="COLOR_NAME", values=metric_col, title=f"Color share for model {selected_model}", hole=0.4)
                st.plotly_chart(fig_color, use_container_width=True)
                st.dataframe(color_dist.rename(columns={metric_col: "value", "pct": "percentage"}))
            else:
                st.info("No color data for this model.")
    else:
        st.info("Color analysis requires MODEL_CODE and COLOR_NAME columns plus QTY or NET_TOTAL.")

    # ----------------------------
    # NEW: Season where each model sells the most
    # ----------------------------
    st.markdown("### Top Season per Model")
    if "MODEL_CODE" in df_selection.columns and "SEASON" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        model_season = df_selection.groupby(["MODEL_CODE", "SEASON"])["NET_TOTAL"].sum().reset_index()
        # find season with max sales per model
        idx = model_season.groupby("MODEL_CODE")["NET_TOTAL"].idxmax()
        top_season_per_model = model_season.loc[idx].reset_index(drop=True).sort_values("NET_TOTAL", ascending=False)
        top_season_per_model = top_season_per_model.rename(columns={"NET_TOTAL": "TOTAL_SALES"})
        st.dataframe(top_season_per_model.head(200))  # show top 200 to keep UI responsive
    else:
        st.info("Season-by-model analysis requires MODEL_CODE, SEASON and NET_TOTAL columns.")

    # ----------------------------
    # NEW: Sales per Season (global)
    # ----------------------------
    st.markdown("### Sales by Season (global)")
    if "SEASON" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        season_sales = df_selection.groupby("SEASON")["NET_TOTAL"].sum().reset_index().sort_values("NET_TOTAL", ascending=False)
        fig_season_bar = px.bar(season_sales, x="SEASON", y="NET_TOTAL", title="Sales by Season (amount)", text="NET_TOTAL")
        st.plotly_chart(fig_season_bar, use_container_width=True)
    else:
        st.info("Seasonal sales require SEASON and NET_TOTAL columns.")

    # ----------------------------
    # NEW: Top 5 models to promote per season (recommendation)
    # ----------------------------
    st.markdown("### Recommended Top Models per Season (Top 5)")
    if "SEASON" in df_selection.columns and "MODEL_CODE" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        season_model = df_selection.groupby(["SEASON", "MODEL_CODE", "CATEGORY"])["NET_TOTAL"].sum().reset_index()
        seasons = season_model["SEASON"].dropna().unique()
        chosen = st.selectbox("Choose season to see recommendations:", options=seasons)
        if chosen:
            top5 = season_model[season_model["SEASON"] == chosen].sort_values("NET_TOTAL", ascending=False).head(5).reset_index(drop=True)
            st.markdown(f"Top 5 recommended models to focus on for **{chosen}** (by sales amount)")
            st.table(top5[["MODEL_CODE", "CATEGORY", "NET_TOTAL"]].assign(Rank=range(1, len(top5)+1)).set_index("Rank"))
            # small bar chart
            fig_top5 = px.bar(top5, x="MODEL_CODE", y="NET_TOTAL", color="CATEGORY", title=f"Top 5 Models for {chosen}")
            st.plotly_chart(fig_top5, use_container_width=True)
            # Practical suggestion text
            st.info(f"Suggestion: prioritize stock/visual merchandising for these models in {chosen} season and consider cross-selling with related categories.")
    else:
        st.info("Recommendation requires SEASON, MODEL_CODE and NET_TOTAL columns.")

    # ----------------------------
    # NEW: For each store, which terminal sells best
    # ----------------------------
    st.markdown("### Best Terminal per Store")
    if "STORE_NAME" in df_selection.columns and "TERMINAL" in df_selection.columns and "NET_TOTAL" in df_selection.columns:
        store_terminal = df_selection.groupby(["STORE_NAME", "TERMINAL"])["NET_TOTAL"].sum().reset_index()
        # find top terminal per store
        idx_term = store_terminal.groupby("STORE_NAME")["NET_TOTAL"].idxmax()
        best_terminal_per_store = store_terminal.loc[idx_term].reset_index(drop=True).sort_values(["STORE_NAME"])
        st.dataframe(best_terminal_per_store.rename(columns={"NET_TOTAL": "TOTAL_SALES"}))
        # allow selection of a store to visualize terminals breakdown
        st.markdown("Terminal breakdown for a selected store")
        store_for_term = st.selectbox("Choose store to see terminal breakdown:", options=best_terminal_per_store["STORE_NAME"].unique())
        if store_for_term:
            breakdown = store_terminal[store_terminal["STORE_NAME"] == store_for_term].sort_values("NET_TOTAL", ascending=False)
            fig_term = px.bar(breakdown, x="TERMINAL", y="NET_TOTAL", title=f"Terminal Sales in {store_for_term}", text="NET_TOTAL")
            st.plotly_chart(fig_term, use_container_width=True)
    else:
        st.info("Terminal analysis requires STORE_NAME, TERMINAL and NET_TOTAL columns.")

else:
    st.warning("Please upload a CSV or Excel file.")
