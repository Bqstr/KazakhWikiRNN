import re

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
        # Process the element here if needed
        pass

    # Check if the event is the end of an element
    elif event == 'end':
        # Check if the element has text
        if element.text:
            # Extract the first 100 characters of the text (if text length is greater than 100)
            extracted_text = element.text
            words =pattern.findall(extracted_text)
            list =list+words
            #print(len(list))
            data = {"word": list}
            print(len(list))
            if(len(list)>140000):
                break

        # Clear the element from memory to free up resources
        element.clear()

df = pd.DataFrame(data)
df.to_csv("myDataset.csv", index=False)



