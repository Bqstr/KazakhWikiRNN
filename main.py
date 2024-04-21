#
import re
import subprocess

























import numpy as np
import sequences
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.src.layers import LSTM
from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.models import Model

import createAndRunRNN_Model


def generate_text(seed_text, next_words, model, max_sequence_len):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        predicted = model.predict_classes(token_list, verbose=0)
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    return seed_text

# Example usage
# seed_text = "The"
# generated_text = generate_text(seed_text, 50, model, seq_length)
# print(generated_text)




createAndRunRNN_Model.run()






































#     # Check if the event is the start of an element
#     if event == 'start':
#
#         # Process the element here if needed
#         pass
#
#     # Check if the event is the end of an element
#     elif event == 'end':
#         # Check if the element has text
#         if element.text:
#             # Extract the first 100 characters of the text (if text length is greater than 100)
#             extracted_text = element.text[:100]
#             #print(extracted_text)
#
#         # Clear the ele=-nt from memory to free up resources
#         element.clear()
#
# # Clean up any remaining references to the root element

#print(len(list))






#
#
# xml_content = """
# Уикипедия kkwiki https://kk.wikipedia.org/wiki/%D0%91%D0%B0%D1%81%D1%82%D1%8B_%D0%B1%D0%B5%D1%82 MediaWiki 1.42.0-wmf.24 first-letter Таспа Арнайы Талқылау Қатысушы Қатысушы талқылауы Уикипедия Уикипедия талқылауы Сурет Сурет талқылауы МедиаУики МедиаУики талқылауы Үлгі Үлгі талқылауы Анықтама Анықтама талқылауы Санат Санат талқылауы Портал Портал талқылауы Жоба Жоба талқылауы TimedText TimedText talk Module Module talk Topic Басты бет 0 1 3302550 3205435 2024-03-29T19:37:14Z Kasymov 10777 wikitext text/x-wiki __NOTOC____NOEDITSECTION__ <templatestyles src="Үлгі:Басты бет/styles.css" /> {{Басты бет/Кіріспе}} <div class="main-wrapper"> <div class="main-wrapper-column"> {{Басты бет/Бөлім |id = tfa |атауы = Таңдаулы мақала |тақырып = {{Басты бет/Таңдаулы|көрсет=сілтеме}} |мазмұн = {{Басты бет/Таңдаулы}} |әрекет = * {{Басты бет/Батырма | {{Басты бет/Таңдаулы|көрсет=тақырып}} | Оқу }} * <span class="nomobile">{{Басты бет/Мәзір | Уикипедия:Таңдаулы мақалалар | Барлық таңдаулы мақала }}</span> |қосалқы = * {{Басты бет/Мәзір | Уикипедия:Таңдаулы мақалаға үміткерлер | Үміткерлер }} {{#if: {{Басты бет/Таңдаулы|көрсет=қосымша}} | : ''Алдыңғылар:'' {{Басты бет/Таңдаулы|көрсет=қосымша}} }} |негізгі сурет = 1 }} {{Басты бет/Бөлім |id = tga |атауы = Жақсы мақала |тақырып = {{Басты бет/Жақсы|көрсет=сілтеме}} |мазмұн = {{Басты бет/Жақсы}} |әрекет = * {{Басты бет/Батырма | {{Басты бет/Жақсы|көрсет=тақырып}} | Оқу }} * <span class="nomobile">{{Басты бет/Мәзір | Уикипедия:Жақсы мақалалар | Барлық жақсы мақала }}</span> |қосалқы = * {{Басты бет/Мәзір | Уикипедия:Жақсы мақалаға үміткерлер | Үміткерлер }} {{#if: {{Басты бет/Жақсы|көрсет=қосымша}} | : ''Алдыңғылар:'' {{Басты бет/Жақсы|көрсет=қосымша}} }} |негізгі сурет = 1 }} {{Басты бет/Бөлім |id = potd |атауы = |тақырып = Тәулік суреті |мазмұн = <div class="main-box-content">[[Сурет:{{trim| {{Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY2}}}} }}|center|500x500px|{{trim| {{Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY2}} (kk)}} }}]]</div> {{#ifexist: Үлгі:Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY2}} (kk) | <div class="main-box-imageCaption"> {{Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY2}} (kk)}} </div> }} |әрекет = |қосалқы = * {{Басты бет/Мәзір | Уикипедия:Тәулік суреті | Үлгіні көру}} {{#ifexist: Үлгі:Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY}} (kk) | | *[[Үлгі:Potd/{{LOCALYEAR}}-{{LOCALMONTH}}-{{LOCALDAY}} (kk)|<span class="mw-ui-button mw-ui-quiet">Сипаттама жазу</span>]]}} |таңдаулы сурет = 1 }} {{Басты бет/Бөлім |id = wmfsp |тақырып = [[Уикимедиа қоры]]ның басқа да [[:meta:Special:MyLanguage/Wikimedia projects|жобалары]] |мазмұн = {{Басты бет/Уикипедияға туыстас жобалар}} }} </div> <div class="main-wrapper-column"> {{Басты бет/Бөлім |id = |атауы = |тақырып = Білгенге маржан |мазмұн = {{Басты бет/Білгенге маржан}} |әрекет = * {{Басты бет/Батырма | Жоба:Білгенге маржан/Ұсыныстар/{{LOCALYEAR}} | Ұсыну }} |қосалқы = * {{Басты бет/Мәзір | Жоба:Білгенге маржан/Мұрағат | Мұрағат}} |негізгі сурет = 1 }} {{Басты бет/Бөлім |id = |атауы = |тақырып = Жаңалықтар |мазмұн = {{Басты бет/Жаңалықтар}} |әрекет = * {{Басты бет/Батырма | Үлгі:Басты бет/Жаңалықтар| Қарау }} |қосалқы = |негізгі сурет = }} {{Басты бет/Бөлім |id = mp-otd |атауы = Күндерек |тақырып = [[{{#time:j F|+06 hours}}]] |мазмұн = {{Уикипедия:Бүгін/{{#time:j F|+06 hours}}}} |әрекет = {{Басты бет/Батырма | {{#time:j F|+06 hours}} | Қарау }} |қосалқы = {{Басты бет/Мәзір| {{#time: j F | {{#time: Y|+5 hours}} -1 days}} | {{#time: j F | {{#time: Y|+6 hours}} -1 days}} }} {{Басты бет/Мәзір| {{#time: j F | {{#time: Y|+5 hours}} +1 days}} | {{#time: j F | {{#time: Y|+5 hours}} +1 days}} }} {{Басты бет/Мәзір | Жыл күндерінің тізімі | Жыл күндерінің тізімі}} |негізгі сурет = }} {{Басты бет/Бөлім |id = wiki |атауы = |тақырып = {{ай атауы бас әріптен|{{#time:F|+5 hours}}}} айына ортақ жұмыс |мазмұн = {{Айдың ортақ жұмысы}} |әрекет = * {{Айдың ортақ жұмысы/Асты}} |қосалқы = * {{Басты бет/Мәзір | Уикипедия:Айдың ортақ жұмысына үміткерлер | Үміткерлер}} * {{Басты бет/Мәзір | Уикипедия:Айдың ортақ жұмысына үміткерлер/Мұрағат | Мұрағат}} |негізгі сурет = 1 }} </div></div> <!-- Интеруикилер --> {{Басты бет/Интеруикилер}}{{noexternallanglinks}}{{nobots}} 5txyif102jf9prkhla2vlq15kienzez МедиаУики:Sitesubtitle 8 75 2297565 25042 2015-04-18T05:37:14Z Arystanbek 14348 wikitext text/x-wiki Қазақша ашық энциклопедия qeuvmnjydumpsskh65tg5ucifedxlvk МедиаУики:Readonlytext 8 98 2508493 2508492 2016-12-06T06:28:00Z Arystanbek 14348 wikitext text/x-wiki {{fmbox |id = mw-readonlytext |type = editnotice |image = [[File:Padlock.svg|45px|link=|Дерекқор құлыпталған]] |text = Төмендегі себеп үшін Уикипедия дерекқоры уақытша тек оқу режимінде: <div>'''$1'''</div> Бұл, бәлкім, күнделікті техникалық қызмет көрсетуге байланысты болуы мүмкін, егер солай болса, сіз бірнеше минуттан кейін қайта өңдей аласыз. Қандай да бір қолайсыздықтар туындаған болса кешірім сұраймыз. Сіз дереккқор құлыпталған кезде Уикипедия мақалаларын шолуды жалғастыра аласыз. Толығырақ ақпарат үшін [http://freenode.net/ freenode] [[Internet Relay Chat|IRC]] желісіндегі [irc://irc.freenode.net/wikipedia #wikipedia] арнасына барып-шыға аласыз. }} c9iz84cycglrya7qmq89jvj59bic10s МедиаУики:Cannotdelete 8 108 2039521 2039519 2014-04-23T17:38:42Z Arystanbek 14348 wikitext text/x-wiki <div class="plainlinks"> «$1» беті немесе файлы жойылмайды. Мұны басқа біреу әлдеқашан [[Special:Log/delete|жойған]] болуы мүмкін. Қайта оралу: *[{{fullurl:Category:Жедел_жоюға_ұсынылғандар|action=purge}} Санат:Жедел жоюға ұсынылғандар] *[[Арнайы:Жаңа беттер]] *[[Арнайы:Жуықтағы өзгерістер]] ([{{fullurl:Special:Recentchanges|hideliu=1&hideminor=1}} тек кірмегендер үшін]). *[[Арнайы:Contributions/newbies|Жаңа тіркелгендер үлесі]] </div>
# """
#
# # Define a regular expression pattern to extract words
# pattern = re.compile(r'\b[а-яәіңғүұқөһ]+[^\W\d_]+\b', re.IGNORECASE)
#
# # Find all words in the text using the pattern
# words = pattern.findall(xml_content)
#
# # Print the list of words
# print(words)