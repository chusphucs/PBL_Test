import requests
import pandas as pd
from bs4 import BeautifulSoup

pageTarget = 'https://pixabay.com/vi/images/search/c%C3%A2y%20b%E1%BA%A1c%20h%C3%A0/'
page = requests.get(pageTarget)
soup = BeautifulSoup(page.content, 'html.parser')
wrapper = soup.find('body')

images = wrapper.find_all("img")
for image in images:
  imgData = image['src']
  print(imgData)
  if("data:image" not in imgData):
    if(imgData):
      downloadPath = './download'
      filename = imgData.split('/')[-1]

      response = requests.get(imgData)

      file = open(downloadPath + filename, "wb")
      file.write(response.content)
      file.close()