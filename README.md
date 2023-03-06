# CLTV Prediction with BG-NBD and Gamma-Gamma Models


1. Data Preperation
2. Expected Number of Transaction with BG-NBD Model
3. Expected Average Profit with Gamma-Gamma Model
4. CLTV Calculations with BG-NBD and Gamma-Gamma Models
5. Creating Segments According to CLTV
6. Functionalization

**Business Problem**: 
An e-commerce company divides its customers into segments and wants to define marketing strategies according to these segments.
Dataset includes the sales of a UK based online store from 01/12/2009 to 09/12/2011.

**Variables**:
- **Invoice:** Unique number for every transaction. Invoices that start with C are cancelled transactions.
- **StockCode:** Unique number for every product.
- **Description:** Product name.
- **Quantity:** Number of specific product that has been ordered in an invoice.
- **InvoiceDate:** Invoice date including the time.
- **Price**: Unit prices(GBP)
- **Customer ID:** Unique number for each customer.
- **Country:** Country the customer lives.

[Medium Blog on this Problem](https://medium.com/@denizcansuturan/customer-lifetime-value-prediction-47bddeaf4174)
