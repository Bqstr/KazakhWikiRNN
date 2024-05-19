import re
import csv

import pandas as pd
import xml.etree.ElementTree as ET
from logging import root

pattern = re.compile(r'\b[а-яәіңғүұқөһ]+[^\W\d_]+\b', re.IGNORECASE)

xml_file = 'kkwiki-20240401-pages-meta-current.xml'
MYTEXT =""
list =[]
count =0
print(ET.iterparse(xml_file))

# Iterate over the XML file incrementally
for event, element in ET.iterparse(xml_file, events=('start', 'end')):
    if event == 'start':
        pass

    elif event == 'end':
        if element.text:
            extracted_text = element.text
            wordss =pattern.findall(extracted_text)
            words = [item + " " for item in wordss]
            print(words)
            list =list+words
            data = {"word": list}
            print(len(list))
            if(len(list)>78000):
                break
        element.clear()

df = pd.DataFrame(data)
df.to_csv("myDataset.csv", index=False)





