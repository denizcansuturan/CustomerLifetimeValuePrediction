##############################################################
# CLTV Prediction BG-NBD and Gamma-Gamma
##############################################################

# 1. Data Preperation
# 2. Expected Number of Transaction with BG-NBD Model
# 3. Expected Average Profit with Gamma-Gamma Model
# 4. CLTV Calculations with BG-NBD and Gamma-Gamma Models
# 5. Creating Segments According to CLTV
# 6. Functionalization

##############################################################
# Data Preperation
##############################################################

# An e-commerce company divides its customers into segments and
# wants to define marketing strategies according to these segments.

# https://archive.ics.uci.edu/ml/datasets/Online+Retail+II

# Dataset includes the sales of a UK based online store
# from 01/12/2009 to 09/12/2011.

# Variables
#
# Invoice: Unique number for every transaction. Invoices that start with C are cancelled transactions.
# StockCode: Unique number for every product.
# Description: Product name.
# Quantity: Number of specific product that has been ordered in an invoice.
# InvoiceDate: Invoice date including the time.
# Price: Unit prices(GBP)
# CustomerID: Unique number for each customer.
# Country: Country the customer lives.


##########################
# Importing necessary libraries and functions
##########################

# !pip install lifetimes
import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions

pd.set_option('display.max_columns', None) # to see all the columns
pd.set_option('display.width', 500) # to see all the columns horizontally
pd.set_option('display.float_format', lambda x: '%.4f' % x) # to adjust decimal points in float numbers
# from sklearn.preprocessing import MinMaxScaler


# Boxplot method:

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    # quartile values are calculated
    interquantile_range = quartile3 - quartile1
    # difference of quartile values is calculated
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit
# this function determines thresholds for the variable.


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#########################
# Reading the data
#########################


df_ = pd.read_excel("D:/MIUUL/CRM/crmAnalytics/rfm/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy() # if something goes wron we do not need to read the data again
                # working on the copy

df.describe().T # numeric variables statistics
df.head() # to get to know the data
df.isnull().sum() # total number of the null values for each variables

df.dropna(inplace=True) # empty customer id data are erased
df = df[~df["Invoice"].str.contains("C", na=False)] # cancelled invoices are erased
df = df[df["Quantity"] > 0] # negative values of Quantity are meaningless and erased
df = df[df["Price"] > 0] # negative values of Price are meaningless and erased
# explained in previous articles.

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")
# outliers are replaced with the limits

df["TotalPrice"] = df["Quantity"] * df["Price"]
# new column is added since it is necessary

df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)
# 2 days later than the max date of the data
# it needs to be cahnged as 2011, 12, 11 for the other sheet of the file

#########################
# Preparing Lifetime Data Structure
#########################

# recency (different from the one in RFM): Time between the first and the last purchase of a customer in weeks customer specific
# T: Age of the customer i.e. time after the first purchase until today in weeks
# frequency: Total number of transactions (frequency>1): (retention)
# monetary: Average monetary per order


cltv_df = df.groupby('Customer ID').agg(
    {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days, # for recency
                     lambda InvoiceDate: (today_date - InvoiceDate.min()).days], # for the age of the customer
     'Invoice': lambda Invoice: Invoice.nunique(),
     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
# droplevel(0) deletes the first row of the column names

cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
# rename the table

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# monetary was the total price in first table, but we need the average
# that is why it is divided by the frequency
cltv_df.head()
cltv_df.describe().T
# to check

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
# retained customers are extracted

cltv_df["recency"] = cltv_df["recency"] / 7
# recency was in days in first table, but we need it in weeks by definition of the models

cltv_df["T"] = cltv_df["T"] / 7
# age was in days in first table, but we need it in weeks by definition of the models
cltv_df.describe().T

##############################################################
# 2. Establishment of BG-NBD Model
##############################################################

# models number of tranactions

bgf = BetaGeoFitter(penalizer_coef=0.001)
# beta and gamma distribution is used. penalizer_coef is needed in this process.
# bgf is the model object

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])
# bgf is fitted

################################################################
# Who are the 10 customers we expect to purchase the most in 1 week?
################################################################

bgf.conditional_expected_number_of_purchases_up_to_time(1,  # means predict for 1 week
                                                        cltv_df['frequency'],
                                                        cltv_df['recency'],
                                                        cltv_df['T']).sort_values(ascending=False).head(10)
# predicts for all the customers, then sorts it descendingly and takes the top 10

bgf.predict(1,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)
# same function with different names and from different libraries
# valid for BG-NBD Model, not for Gamma Gamma Model

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])
# expected purchase in 1 week is added for all the customers

################################################################
# Who are the 10 customers we expect to purchase the most in 1 month?
################################################################

bgf.predict(4, # in weeks, equals to a month
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

bgf.predict(4,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

################################################################
# What is the Expected Number of Sales of the Whole Company in 3 Months?
################################################################

bgf.predict(4 * 3,
            cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T']).sum()

cltv_df["expected_purc_3_month"] = bgf.predict(4 * 3,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.head()
################################################################
# Evaluation of Prediction Results
################################################################

plot_period_transactions(bgf)
plt.show()

##############################################################
# 3. Establishment of GAMMA-GAMMA Model
##############################################################

# models expected_average_profit

ggf = GammaGammaFitter(penalizer_coef=0.01)
# ggf is the model object
# the object finds the parameters of the model

ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
# ggf is fitted
#  <lifetimes.GammaGammaFitter: fitted with 2893 subjects, p: 3.79, q: 0.34, v: 3.73>

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).head(10)

ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
cltv_df.sort_values("expected_average_profit", ascending=False).head(10)

##############################################################
# 4. Calculation of CLTV with BG-NBD and GG model.
##############################################################

# 2 models are merged
cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=3,  # in months by definition of the function
                                   freq="W",  # T frequency info (in weeks)
                                   discount_rate=0.01) # if there is a discount

cltv.head()

cltv = cltv.reset_index()
# customer id is transformed from index to a column

cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
# merge function is introduced, merging 2 dataframes on Customer ID with left join

cltv_final.sort_values(by="clv", ascending=False).head(10)
# customer_lifetime_value function returns cltv as clv by default

##############################################################
# 5. Segmentation According to CLTV
##############################################################

cltv_final

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

cltv_final.sort_values(by="clv", ascending=False).head(5)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})



##############################################################
# 6. Functionalization
##############################################################

def create_cltv_p(dataframe, month=3):
    # 1. Data Preparation
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]
    dataframe = dataframe[dataframe["Quantity"] > 0]
    dataframe = dataframe[dataframe["Price"] > 0]
    replace_with_thresholds(dataframe, "Quantity")
    replace_with_thresholds(dataframe, "Price")
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    today_date = dt.datetime(2011, 12, 11)
    # 2010 for first dataset

    cltv_df = dataframe.groupby('Customer ID').agg(
        {'InvoiceDate': [lambda InvoiceDate: (InvoiceDate.max() - InvoiceDate.min()).days,
                         lambda InvoiceDate: (today_date - InvoiceDate.min()).days],
         'Invoice': lambda Invoice: Invoice.nunique(),
         'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
    cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
    cltv_df["recency"] = cltv_df["recency"] / 7
    cltv_df["T"] = cltv_df["T"] / 7

    # 2. BG-NBD Model Establishment
    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency'],
            cltv_df['T'])

    cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                                  cltv_df['frequency'],
                                                  cltv_df['recency'],
                                                  cltv_df['T'])

    cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    cltv_df["expected_purc_3_month"] = bgf.predict(12,
                                                   cltv_df['frequency'],
                                                   cltv_df['recency'],
                                                   cltv_df['T'])

    # 3. GAMMA-GAMMA Model Establishment
    ggf = GammaGammaFitter(penalizer_coef=0.01)
    ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
    cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                                 cltv_df['monetary'])

    # 4. CLTV Calculation with BG-NBD and GG models
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency'],
                                       cltv_df['T'],
                                       cltv_df['monetary'],
                                       time=month,  # 3 aylık
                                       freq="W",  # T'nin frekans bilgisi.
                                       discount_rate=0.01)

    cltv = cltv.reset_index()
    cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels=["D", "C", "B", "A"])

    return cltv_final


df2_ = pd.read_excel("D:/MIUUL/CRM/crmAnalytics/rfm/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df2 = df2_.copy()


cltv_final2 = create_cltv_p(df2)
cltv_final2.sort_values(by="clv", ascending=False).head(5)
cltv_final2.to_csv("cltv_prediction.csv")














