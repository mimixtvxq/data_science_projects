#!/usr/bin/env python
# coding: utf-8

# In[187]:


# -*- coding: utf-8 -*-
#Tanasorn Chindasook 
#Web crawler + Google Sheets updater for apartment hunting 

from bs4 import BeautifulSoup as soup
import pandas as pd
import requests
from pprint import pprint
import sys

#GoogleSheets API
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# In[188]:


args = sys.argv


# In[189]:


# GOOGLE SHEET KEY
GSKEY = '1ffjHfhzP9IHINpNJcKBOsDLRA2lSWA5VrcreDNe-o30'


# In[190]:


# Connect to Google Sheets 

scope = ["https://spreadsheets.google.com/feeds",
         'https://www.googleapis.com/auth/spreadsheets',
         "https://www.googleapis.com/auth/drive.file",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name('credentials.json',scope)
client = gspread.authorize(creds)

sheet = client.open_by_key(GSKEY).sheet1


# In[250]:


# fetch URLS from the other google sheets
list_of_urls = client.open_by_key(GSKEY).worksheet('Sheet2')

urls = list_of_urls.get_all_records()

URLS = []
for url in urls:
    URLS.append(list(url.values()))

# URL or List of URLS
URLS = [item for sublist in URLS for item in sublist]


# In[255]:


# BeautifulSoup HTML retrieval
for URL in URLS:
    r = requests.get(URL, verify=False)
    page_html = r.text
    page_soup = soup(page_html,'html.parser')

    # Get variables from webpage to insert
    flat_name = page_soup.find('h1').text
    try:
        address_1 = page_soup.find('span',{'class':'block font-nowrap print-hide'}).text
    except:
        address_1 = ''
    try:
        address_2 = page_soup.find('span',{'class':'zip-region-and-country'}).text
    except:
        address_2 = ''
    address = (address_1 + address_2).strip()

    try:
        rooms = int(page_soup.find('div',{'class':'is24qa-zi is24-value font-semibold'}).text.strip())
    except:
        rooms = ''
    try:
        sqm = float(page_soup.find('div',{'class':'is24qa-flaeche is24-value font-semibold'})    .text.strip()    .replace(',','.')    .split(' ')    [0])
    except:
        sqm = None
    try:
        move_in_date = page_soup.find('dd',{'class':'is24qa-bezugsfrei-ab grid-item three-fifths'}).text.strip()
    except:
        move_in_date = ''
        
    try:
        floor = page_soup.find('dd',{'class':'is24qa-etage grid-item three-fifths'}).text.strip()
    except:
        floor = 0
    try:
        price_warm = float(page_soup.find('dd',{'class':'is24qa-gesamtmiete grid-item three-fifths font-bold'})        .text.strip()        .replace('.','')        .split(' ')        [0])
    except:
        price_warm = 0
    try:
        pps = (price_warm)/float(sqm)
    except:
        pps = None
    
    # create row to update in gsheets
    insert_row = [flat_name, sqm, rooms, floor, price_warm, address, URL, '', '', '','','','','',pps, move_in_date]
    
    data = sheet.get_all_records()
    
    #insert row into gsheets
    sheet.insert_row(insert_row,len(data) + 2)






