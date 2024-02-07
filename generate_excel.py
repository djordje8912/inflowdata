import xlsxwriter
import random
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
import numpy as np
import os

#from win32com import client
#import pdfkit

df = pd.read_csv('AllDetails.csv')
with open("InflowForecast.dll.config", "r") as file:
    # Read each line in the file, readlines() returns a list of lines
    content = file.readlines()
    # Combine the lines in the list into a string
    content = "".join(content)
    #print(content)
    bs_content = BeautifulSoup(content)
    CSV_location=bs_content.find("add", {"key": "locationOfcsv"}).get('value')  
    archive_excel=bs_content.find("add", {"key": "archive_excel"}).get('value')  
    locationOfExcel=bs_content.find("add", {"key": "locationOfExcel"}).get('value')  
    resolution=bs_content.find("add", {"key": "resolution"}).get('value')  
    startOfPrediction=bs_content.find("add", {"key": "startOfPrediction"}).get('value')  
    locationOfArchiveExcel=bs_content.find("add", {"key": "locationOfArchiveExcel"}).get('value')  
    openExcel=bs_content.find("add", {"key": "open_excel"}).get('value')  

df['Date'] = pd.to_datetime(df['Date'])
datetimeStartOfPrediction = datetime.strptime(startOfPrediction, '%Y-%m-%dT%H:%M')
if(resolution=="1440"):
    df=df[df['Date'] >= datetimeStartOfPrediction.date()] 
else:
    df=df[df['Date'] >= datetimeStartOfPrediction]
# Example data
# Try to do as much processing outside of initializing the workbook
# Everything beetween Workbook() and close() gets trapped in an exception
#random_data = [random.random() for _ in range(10)]
Date=df["Date"].dt.strftime('%d-%m-%Y %H:%M')
Inflows=df["Inflows"]
InflowsTributary=df["InflowsTributary"]
InflowsTributary2=df["InflowsTributary2"]

Precipitation1=df["Fierze_rain"]
Precipitation2=df["Koman_rain"]
Precipitation3=df["VauDejes_rain"]

Temperature1=df["Fierze_temp"]
Temperature2=df["Koman_temp"]
Temperature3=df["VauDejes_temp"]

Humidity1=df["Fierze_humi"]
Humidity2=df["Koman_humi"]
Humidity3=df["VauDejes_humi"]


# Data location inside excel

data_start_loc_precipitation = [2,2]
data_end_loc_precipitation = [data_start_loc_precipitation[0] + len(df.index),2]
data_start_loc_precipitation1 = [2,5]
data_end_loc_precipitation1 = [data_start_loc_precipitation1[0] + len(df.index),5]
data_start_loc_precipitation2= [2,8]
data_end_loc_precipitation2 = [data_start_loc_precipitation2[0] + len(df.index),8]


data_start_loc_temperature = [2,1]
data_end_loc_temperature = [data_start_loc_temperature[0] + len(df.index),1]
data_start_loc_temperature1 = [2,4]
data_end_loc_temperature1 = [data_start_loc_temperature1[0] + len(df.index),4]
data_start_loc_temperature2= [2,7]
data_end_loc_temperature2 = [data_start_loc_temperature2[0] + len(df.index),7]

data_start_loc_humidity = [2,3]
data_end_loc_humidity = [data_start_loc_humidity[0] + len(df.index),3]
data_start_loc_humidity1 = [2,6]
data_end_loc_humidity1 = [data_start_loc_humidity1[0] + len(df.index),6]
data_start_loc_humidity2= [2,9]
data_end_loc_humidity2 = [data_start_loc_humidity2[0] + len(df.index),9]

data_start_loc_date = [2,0] # xlsxwriter rquires list, no tuple


data_start_loc_inflows = [2,1]
data_start_loc_inflowstributary = [2,2]
data_start_loc_inflowstributary2= [2,3]














data_end_loc_inflows = [data_start_loc_inflows[0] + len(df.index),1]
data_end_loc_inflowstributary = [data_start_loc_inflows[0] + len(df.index),2]
data_end_loc_inflowstributary2 = [data_start_loc_inflows[0] + len(df.index),3]
workbook = xlsxwriter.Workbook(locationOfExcel)



worksheet = workbook.add_worksheet("Weather forecast")


# Increase the cell size of the merged cells to highlight the formatting.
worksheet.set_column("A:A", 25)
worksheet.set_column("B:J", 17)
worksheet.set_row(0, 30)
worksheet.set_row(1, 20)



# Create a format to use in the merged range.
date_format = workbook.add_format(
    {
        "bold": 1,
        "border": 1,
        "align": "center",
        "valign": "vcenter"
    }
)
fierze_format = workbook.add_format(
    {
        "bold": 1,
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "fg_color": "red",
    }
)
koman_format = workbook.add_format(
    {
        "bold": 1,
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "fg_color": "green",
    }
)
vaudejes_format = workbook.add_format(
    {
        "bold": 1,
        "border": 1,
        "align": "center",
        "valign": "vcenter",
        "fg_color": "yellow",
    }
)
values_format = workbook.add_format(
    {
        
        "align": "center",
        "valign": "vcenter"
       
    }
)

#worksheet.set_row('2:500', cell_format=values_format)

# Merge 3 cells.
worksheet.merge_range("A1:A2", "DATE",date_format)
worksheet.merge_range("B1:D1", "Fierze", fierze_format)


worksheet.merge_range("E1:G1", "Koman", koman_format)
worksheet.merge_range("H1:J1", "Vau Dejes", vaudejes_format)


# A chart requires data to reference data inside excel

worksheet.write(1, 1, "temperature [°C]") 
worksheet.write(1, 2, "preciptation [mm]") 
worksheet.write(1, 3, "humidity [%]") 
worksheet.write(1, 4, "temperature [°C]") 
worksheet.write(1, 5, "preciptation [mm]") 
worksheet.write(1, 6, "humidity [%]") 
worksheet.write(1, 7, "temperature [°C]") 
worksheet.write(1, 8, "preciptation [mm]") 
worksheet.write(1, 9, "humidity [%]") 


#worksheet.set_column(0, 0, 20) 

worksheet.write_column(*data_start_loc_precipitation, data=Precipitation1)
worksheet.write_column(*data_start_loc_precipitation1, data=Precipitation2)
worksheet.write_column(*data_start_loc_precipitation2, data=Precipitation3)

worksheet.write_column(*data_start_loc_temperature, data=Temperature1)
worksheet.write_column(*data_start_loc_temperature1, data=Temperature2)
worksheet.write_column(*data_start_loc_temperature2, data=Temperature3)

worksheet.write_column(*data_start_loc_humidity, data=Humidity1)
worksheet.write_column(*data_start_loc_humidity1, data=Humidity2)
worksheet.write_column(*data_start_loc_humidity2, data=Humidity3)

worksheet.write_column(*data_start_loc_date, data=Date)










# Charts are independent of worksheets
chart2 = workbook.add_chart({'type': 'column'})
chart2.set_y_axis({'name': 'precipitation[mm]'})
chart2.set_x_axis({'name': 'time'})
chart2.set_title({'name': 'Forecast for precipitation '})
# The chart needs to explicitly reference data
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_precipitation + data_end_loc_precipitation,  
    'name': "Precipitation Fierze",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_precipitation1 + data_end_loc_precipitation1,  
    'name': "Precipitation Koman",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_precipitation2 + data_end_loc_precipitation2,  
    'name': "Precipitation Vau Dajes",
})
worksheet.insert_chart('L'+str(2), chart2, {'x_scale':2.2, 'y_scale': 1.5})

# Charts are independent of worksheets
chart2 = workbook.add_chart({'type': 'line'})
chart2.set_y_axis({'name': 'temperature[°C]'})
chart2.set_x_axis({'name': 'time'})
chart2.set_title({'name': ' Forecast for temperature'})
# The chart needs to explicitly reference data
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_temperature + data_end_loc_temperature,  
    'name': "Temperature Fierze",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_temperature1 + data_end_loc_temperature1,  
    'name': "Temperature Koman",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_temperature2 + data_end_loc_temperature2,  
    'name': "Temperature Vau Dajes",
})
worksheet.insert_chart('L'+str(27), chart2, {'x_scale':2.2, 'y_scale': 1.5})

# Charts are independent of worksheets
chart2 = workbook.add_chart({'type': 'line'})
chart2.set_y_axis({'name': 'humidity[%]'})
chart2.set_x_axis({'name': 'time'})
chart2.set_title({'name': ' Forecast for humidity'})
# The chart needs to explicitly reference data
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_humidity + data_end_loc_humidity,  
    'name': "Humidity Fierze",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_humidity1 + data_end_loc_humidity1,  
    'name': "Humidity Koman",
})
chart2.add_series({
    'values': [worksheet.name] + data_start_loc_humidity2 + data_end_loc_humidity2,  
    'name': "Humidity Vau Dajes",
})
worksheet.insert_chart('L'+str(53), chart2, {'x_scale':2.2, 'y_scale': 1.5})

worksheet = workbook.add_worksheet("Inflow  forecast")
worksheet.set_row(0, 30)
worksheet.set_row(1, 20)
worksheet.set_column("A:A", 25)
worksheet.set_column("B:D", 17)
worksheet.write(0, # <-- The cell row (zero indexed).
                1,  # <-- The cell column (zero indexed).
                "Fierze", fierze_format)
worksheet.write(0, 2,  "Koman", koman_format)
worksheet.write(0, 3,  "Vau Dejes", vaudejes_format)

worksheet.merge_range("A1:A2", "DATE",date_format)

""" worksheet.merge_range("B1:B1", "Fierze", fierze_format)
worksheet.merge_range("C1:C1", "Koman", koman_format)
worksheet.merge_range("D1:D1", "Vau Dejes", vaudejes_format) """

worksheet.merge_range("B2:D2", "inflows[m^3/s]",values_format)


# A chart requires data to reference data inside excel



worksheet.write_column(*data_start_loc_date, data=Date)
worksheet.write_column(*data_start_loc_inflows, data=Inflows)
worksheet.write_column(*data_start_loc_inflowstributary, data=InflowsTributary)
worksheet.write_column(*data_start_loc_inflowstributary2, data=InflowsTributary2) 

# Charts are independent of worksheets
chart = workbook.add_chart({'type': 'line'})
chart.set_y_axis({'name': 'Inflow[m^3/s]'})
chart.set_x_axis({'name': 'time'})
chart.set_title({'name': 'Inflows forecast'})
# The chart needs to explicitly reference data
chart.add_series({
    'values': [worksheet.name] + data_start_loc_inflows + data_end_loc_inflows,  
    'name': "Inflow Fierze",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_inflowstributary + data_end_loc_inflowstributary,  
    'name': "Inflow Koman",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_inflowstributary2 + data_end_loc_inflowstributary2,  
    'name': "Inflow Vau Dajes",
})
worksheet.insert_chart('F'+str(5), chart, {'x_scale':2.7, 'y_scale': 1.5})





month_dry = pd.read_csv('measurments/dry_year.csv')
month_wet = pd.read_csv('measurments/wet_year.csv')
month_char = pd.read_csv('measurments/characteristic_year.csv')



currentMonth = datetime.now().month
month_dry['Datetime'] =  pd.to_datetime(month_dry['Datetime'], errors='coerce', dayfirst=True)
month_wet['Datetime'] =  pd.to_datetime(month_wet['Datetime'],errors='coerce', dayfirst=True)
month_char['Datetime'] =  pd.to_datetime(month_char['Datetime'],errors='coerce', dayfirst=True)
month_dry = month_dry[month_dry['Datetime'].dt.month == currentMonth]
month_wet = month_wet[month_wet['Datetime'].dt.month == currentMonth]
month_char = month_char[month_char['Datetime'].dt.month == currentMonth]
month_dry=month_dry.reset_index()
date=month_dry.index+1

dry1=month_dry["1_FierzaStorage"]
wet1=month_wet["1_FierzaStorage"]
char1=month_char["1_FierzaStorage"]
dry2=month_dry["2_KomanStorage"]
wet2=month_wet["2_KomanStorage"]
char2=month_char["2_KomanStorage"]
dry3=month_dry["3_VDejesStorage"]
wet3=month_wet["3_VDejesStorage"]
char3=month_char["3_VDejesStorage"]
worksheet = workbook.add_worksheet("Historical statistics")




data_start_loc_date = [2,0] # xlsxwriter rquires list, no tuple
data_start_loc_a1 =[ 2,1]
data_start_loc_d1 =[ 2,2]
data_start_loc_w1= [ 2,3]
data_start_loc_a2 =[ 2,4]
data_start_loc_d2 =[ 2,5]
data_start_loc_w2= [ 2,6]
data_start_loc_a3 =[ 2,7]
data_start_loc_d3 =[ 2,8]
data_start_loc_w3= [ 2,9]
data_end_loc_a1 = [2+ len(month_dry.index),1]
data_end_loc_d1 = [2+ len(month_dry.index),2]
data_end_loc_w1 = [ 2+len(month_dry.index),3]
data_end_loc_a2 = [ 2+len(month_dry.index),4]
data_end_loc_d2 = [ 2+len(month_dry.index),5]
data_end_loc_w2 = [ 2+len(month_dry.index),6]
data_end_loc_a3 = [ 2+len(month_dry.index),7]
data_end_loc_d3 = [ 2+len(month_dry.index),8]
data_end_loc_w3 = [ 2+len(month_dry.index),9]


worksheet.set_column("A:A", 25)
worksheet.set_column("B:J", 17)
worksheet.set_row(0, 30)
worksheet.set_row(1, 20)


worksheet.merge_range("A1:A2", "DATE",date_format)
worksheet.merge_range("B1:D1", "Fierze", fierze_format)


worksheet.merge_range("E1:G1", "Koman", koman_format)
worksheet.merge_range("H1:J1", "Vau Dejes", vaudejes_format)
worksheet.write(1, 1, "average") 
worksheet.write(1, 2, "dry") 
worksheet.write(1, 3, "wet") 
worksheet.write(1, 4, "average") 
worksheet.write(1, 5, "dry") 
worksheet.write(1, 6, "wet") 
worksheet.write(1, 7, "average") 
worksheet.write(1, 8, "dry") 
worksheet.write(1, 9, "wet") 

worksheet.write_column(*data_start_loc_date, data=date)
worksheet.write_column(*data_start_loc_d1, data=dry1)
worksheet.write_column(*data_start_loc_w1, data=wet1)
worksheet.write_column(*data_start_loc_a1, data=char1)
worksheet.write_column(*data_start_loc_d2, data=dry2)
worksheet.write_column(*data_start_loc_w2, data=wet2)
worksheet.write_column(*data_start_loc_a2, data=char2)
worksheet.write_column(*data_start_loc_d3, data=dry3)
worksheet.write_column(*data_start_loc_w3, data=wet3)
worksheet.write_column(*data_start_loc_a3, data=char3)

# Charts are independent of worksheets
chart = workbook.add_chart({'type': 'line'})
chart.set_y_axis({'name': 'inflow[m^3/s]'})
chart.set_x_axis({'name': 'hours'})
chart.set_title({'name': 'Fierze current month statistics'})
# The chart needs to explicitly reference data
chart.add_series({
    'values': [worksheet.name] + data_start_loc_d1 + data_end_loc_d1,  
    'name': " Fierze dry",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_w1 + data_end_loc_w1,  
    'name': "Fierze wet",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_a1 + data_end_loc_a1,  
    'name': "Fierze characteristic",
})
worksheet.insert_chart('M'+str(2), chart, {'x_scale':2.7, 'y_scale': 1.5})



# Charts are independent of worksheets
chart = workbook.add_chart({'type': 'line'})
chart.set_y_axis({'name': 'inflow[m^3/s]'})
chart.set_x_axis({'name': 'hours'})
chart.set_title({'name': 'Koman current month statistics'})
# The chart needs to explicitly reference data
chart.add_series({
    'values': [worksheet.name] + data_start_loc_d2 + data_end_loc_d2,  
    'name': " Koman dry",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_w2 + data_end_loc_w2,  
    'name': "Koman wet",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_a2 + data_end_loc_a2,  
    'name': "Koman characteristic",
})
worksheet.insert_chart('M'+str(27), chart, {'x_scale':2.7, 'y_scale': 1.5})


# Charts are independent of worksheets
chart = workbook.add_chart({'type': 'line'})
chart.set_y_axis({'name': 'inflow[m^3/s]'})
chart.set_x_axis({'name': 'hours'})
chart.set_title({'name': 'VauDejes current month statistics'})
# The chart needs to explicitly reference data
chart.add_series({
    'values': [worksheet.name] + data_start_loc_d3 + data_end_loc_d3,  
    'name': " VauDejes dry",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_w3 + data_end_loc_w3,  
    'name': "VauDejes wet",
})
chart.add_series({
    'values': [worksheet.name] + data_start_loc_a3 + data_end_loc_a3,  
    'name': "VauDejes characteristic",
})
worksheet.insert_chart('M'+str(52), chart, {'x_scale':2.7, 'y_scale': 1.5})

workbook.close()
if(archive_excel.lower()=='true'):
    workbook = xlsxwriter.Workbook(locationOfArchiveExcel+"\\excel_"+datetime.now().strftime("%Y%m%d%H%M%S")+".xslx")
    
  # Write to file
workbook.close()

df = pd.read_csv('AllDetails.csv')
cols_to_keep = ['Date', 'Inflows', 'InflowsTributary','InflowsTributary2']
df=df[cols_to_keep]
df = df.rename(columns={'Inflows': '1_FierzaStorage', 'InflowsTributary': '2_KomanStorage','Date': 'DATETIME', 'InflowsTributary2': '3_VDejesStorage'})
# df['DATETIME'] = df['DATETIME'].dt.strftime('%Y/%m/%d %H:%M:%S')
df=df.iloc[96:]
df.to_csv(CSV_location, index=False) 

if(openExcel.lower()=='true'):
    os.startfile(locationOfExcel)




# Open Microsoft Excel
    #from win32com import client
#excel = client.Dispatch("Excel.Application")
  
# Read Excel File
#sheets = excel.Workbooks.Open('locationOfExcel')
#work_sheets = sheets.Worksheets[1]
  
# Convert into PDF File
#work_sheets.ExportAsFixedFormat(0,'MyPDF.pdf')
    

#df = pd.read_excel(locationOfExcel)#input
#df.to_html("file.html")#to html
#pdfkit.from_file("file.html", "file.pdf")#to pdf