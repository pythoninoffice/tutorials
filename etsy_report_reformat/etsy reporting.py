import pathlib
import pandas as pd

etsy = pathlib.Path(r'folder for your esty reports')
files = etsy.iterdir()
df = pd.DataFrame()

for f in files:
    if f.name.endswith('.csv'):
        df = df.append(pd.read_csv(f), ignore_index = True)
    

df['Amount'].replace('--','CA$0',inplace=True)
df['Fees & Taxes'].replace('--','CA$0',inplace=True)

df['Amount'] = df['Amount'].str.split('$',expand=True)[1].astype(float)
df['Fees & Taxes'] = df['Fees & Taxes'].str.split('$',expand=True)[1].astype(float)
df.to_excel(etsy/'detailed_etsy_report_2022.xlsx', index=False)

summary = df.groupby('Type').agg({'Amount':'sum','Fees & Taxes':'sum'})
summary.loc['total'] = summary.sum(numeric_only=True, axis = 0)

summary.to_excel(etsy/'etsy_report_2022.xlsx')
