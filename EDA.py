import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from scipy.stats import chi2_contingency
import statsmodels.stats.proportion as smp
from statsmodels.formula.api import ols
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import f_classif
import statsmodels.api as sm

def interpret_p_value(p):
    return "Significant" if p < 0.05 else "Not Significant"


def comparative_barplot(data, column, title, data_type='Count'):
    # Filtering data for each region
    data_eu = data[data['region'] == 'EU']
    data_us = data[data['region'] == 'US']

    if data_type == 'Percentage':
        # Calculate percentages for each category within each region
        count_eu = data_eu[column].value_counts(normalize=True).sort_index() * 100
        count_us = data_us[column].value_counts(normalize=True).sort_index() * 100
        y_label = "Percentage (%)"
    else:
        # Use counts
        count_eu = data_eu[column].value_counts().sort_index()
        count_us = data_us[column].value_counts().sort_index()
        y_label = "Count"

    # Creating barplots
    fig = go.Figure(data=[
        go.Bar(name='EU', x=count_eu.index, y=count_eu.values),
        go.Bar(name='US', x=count_us.index, y=count_us.values)
    ])
    fig.update_layout(
        title_text=title, 
        barmode='group',
        yaxis=dict(title=y_label)
    )
    return fig


def calculate_rates(data, column):
    grouped = data.groupby('region')[column]
    counts = grouped.sum()
    total = grouped.count()
    ci_lower, ci_upper = smp.proportion_confint(counts, total, alpha=0.05, method='wilson')
    return counts, total, ci_lower, ci_upper





# Loading data

@st.cache_data()
def load_data(filepath):
    data = pd.read_csv(filepath)
    data['is_dropout'] = data['a_drop_out'] == 1
    opt_out_conditions = ['consent declined', 'camera declined (main: denied, detailed: not-allowed-error)']
    data['is_opt_out'] = data['a_task_completion_reason'].isin(opt_out_conditions)
    data['is_fraud'] = data['high low fraud'] == 'Fraudulent'
    data['region'] = data['region'].replace({'eu-west-1': 'EU', 'us-east-1': 'US'})
    data['a_local_time'].replace('', np.nan, inplace=True)
    data['time_of_day'] = pd.to_datetime(data['a_local_time'], errors='coerce' , utc=True).dt.hour
    
    return data





data = load_data('Panelist_Details_1702378286686.csv')

# ANOVA model test



st.sidebar.title("Filters")
display_type = st.sidebar.radio("Display Type", ["Count", "Percentage"])
column_to_plot = st.sidebar.selectbox("Select Column to Compare", options=["a_os_name","Reported gender","Quality assignement","a_age","a_platform_type","a_qa_duplicate","time_of_day"])

age_range = st.sidebar.slider("Select Age Range", min_value=int(data['a_age'].min()), max_value=int(data['a_age'].max()), value=(25, 75))
analysis_type = st.sidebar.selectbox("Choose analysis type", ["Drop-outs", "Opt-outs"])



# Creating a contingency table
contingency_table = pd.crosstab(data['region'], data['is_fraud'])

chi2, p, dof, expected = chi2_contingency(contingency_table)

# Display the results
st.write("Chi-Square Statistic:", chi2)
st.write("P-value:", p)

# Interpretation
if p < 0.05:
    st.write("Result: Significant difference in fraud proportions between regions.")
else:
    st.write("Result: No significant difference in fraud proportions between regions.")

if display_type == "Percentage":
    # Calculate percentages for each category within each region
    total_counts = contingency_table.sum(axis=1)
    percentage_table = contingency_table.div(total_counts, axis=0) * 100
    fraud_dist_fig = px.bar(percentage_table.T, barmode='group', title="Fraud Distribution by Region (%)",labels={'value': 'rate'})
else:
    # Display counts
    fraud_dist_fig = px.bar(contingency_table.T, barmode='group', title="Fraud Distribution by Region (Count)")

st.plotly_chart(fraud_dist_fig)

# Sidebar for filter options

# Main page
st.title("Comparative Analysis by Region")



# Filtering data
filtered_data = data[(data['a_age'] >= age_range[0]) & (data['a_age'] <= age_range[1])]
bins = [12, 18, 35, 60, 85]  
labels = ['12-18', '18-35', '35-60', '60-85']
if 'age_group' not in filtered_data.columns:
    filtered_data['age_group'] = pd.cut(filtered_data['a_age'], bins=bins, labels=labels, right=False)

if display_type == "Percentage":
    age_dist = (filtered_data.groupby(['region', 'age_group']).size() / filtered_data.groupby(['region']).size()) * 100
    age_dist = age_dist.reset_index(name='Percentage')
    age_fig = px.bar(age_dist, x='age_group', y='Percentage', color='region', barmode='group')
else:
    age_dist = filtered_data.groupby(['region', 'age_group']).size().reset_index(name='Count')
    age_fig = px.bar(age_dist, x='age_group', y='Count', color='region', barmode='group')

age_fig.update_layout(title="Age Group Distribution by Region")
st.plotly_chart(age_fig)


# Dropots, Opt outs rates
if analysis_type == "Drop-outs":
    counts, total, ci_lower, ci_upper = calculate_rates(data, 'is_dropout')
else:
    counts, total, ci_lower, ci_upper = calculate_rates(data, 'is_opt_out')

# Creating DataFrame for visualization
ci_df = pd.DataFrame({
    'Region': counts.index,
    'Rate': (counts / total) * 100,  
    'Lower CI': ci_lower * 100,       
    'Upper CI': ci_upper * 100        
})
fig = go.Figure(data=[
    go.Bar(
        name=f'{analysis_type} Rate',
        x=ci_df['Region'],
        y=ci_df['Rate'],
        error_y=dict(type='data', 
                     array=ci_df['Upper CI'] - ci_df['Rate'], 
                     arrayminus=ci_df['Rate'] - ci_df['Lower CI'])
    )
])


fig.update_layout(title=f'{analysis_type} Rates with Confidence Intervals', xaxis_title='Region', yaxis_title='Rate')
st.plotly_chart(fig)


# Assume other necessary variables are defined (like 'data' and 'column')
fig = comparative_barplot(data, column_to_plot, 'Comperative analysis', data_type=display_type)
st.plotly_chart(fig)

# ANOVA TEST RESULTS 


categorical_columns = ['a_os_name', 'Reported gender','a_platform_type','region']



encoder = LabelEncoder()
for col in categorical_columns:
    data[col] = encoder.fit_transform(data[col])



X = data[['is_opt_out', 'is_dropout', 'a_os_name', 'Reported gender','a_platform_type']]

bool_columns = X.select_dtypes(include=['bool']).columns
X[bool_columns] = X[bool_columns].astype(int)

X = sm.add_constant(X) 
X['const'] = X['const'].astype(int)
print(X.dtypes)


y = data['region']

# Fit the model
model = sm.Logit(y, X).fit()
model_summary = model.summary()
with st.expander("See stats"):
    st.text(str(model_summary))
