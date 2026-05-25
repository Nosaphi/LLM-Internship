import os
from os import listdir
from os.path import isfile, join
import json
from pypdf import PdfReader
import fitz
from pdf2image import convert_from_path
import tempfile
import pytesseract
import spacy
import re
import pandas as pd
import cv2
import numpy as np
import unicodedata
from ftfy import fix_text
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
# Get the absolute path to working directory and setting relative path from here to json files
path = os.getcwd()
dir="/media/protocolli/"

# NER model for anonymization
nlp = spacy.load("it_core_news_lg", disable=["parser", "lemmatizer"])

# Mapping dictionnary for the anonymization
mapping = {
    "EMAIL": {},
    "PERSON": {},
    "PHONE": {},
    "ID": {},
    "ADDRESS": {}
}

KEYWORDS = {"nome", "cognome", "sig", "sig.", "signor", "signora", "monsignor", "presidente"}

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

mostCommonNames = [
    "Abis", "Achenza", "Agus", "Alba", "Ambu", "Angioy", "Angioni", "Anedda", 
    "Aresti", "Argiolas", "Argiu", "Arixi", "Aru", "Asara", "Asquer", 
    "Atzeni", "Atzeni", "Bacciu", "Badas", "Ballocco", "Baris", "Barmina", 
    "Basciu", "Bellu", "Berria", "Biddau", "Boi", "Bolognesi", "Bua", 
    "Brundu", "Cabitza", "Cabras", "Cadeddu", "Caddeo", "Calaresu", "Calia", 
    "Camboni", "Campus", "Canu", "Cappai", "Capula", "Carboni", "Cardia", 
    "Carta", "Casu", "Cau", "Cherchi", "Chessa", "Cilloco", "Cocco", 
    "Coccole", "Collu", "Congiu", "Contini", "Cordeddu", "Corona", "Cossu", 
    "Craba", "Cucca", "Cuccu", "Cugusi", "Curreli", "Daga", "Deiana", 
    "Deidda", "Deligia", "Delogu", "Demelas", "Demontis", "Deriu", "Dessì", 
    "Dettori", "Deu", "Enna", "Erriu", "Fadda", "Falchi", "Farina", 
    "Farris", "Filigheddu", "Floris", "Fodde", "Fois", "Frongia", "Gessa", 
    "Gattu", "Giacobbe", "Giua", "Gungui", "Ibba", "Illotto", "Lai", 
    "Ladu", "Lecis", "Lecca", "Ledda", "Ligas", "Lilliu", "Litarru", 
    "Lizzeri", "Loi", "Loni", "Lorigiga", "Lorrai", "Lovicu", "Macis", 
    "Madau", "Manca", "Manconi", "Manis", "Marini", "Marongiu", "Marras", 
    "Masia", "Masala", "Mascia", "Matta", "Mattu", "Medde", "Melis", 
    "Meloni", "Mereu", "Mesina", "Mele", "Milia", "Moi", "Monni", 
    "Montisci", "Moro", "Moxeddu", "Mulas", "Mura", "Murru", "Muscas", 
    "Musio", "Naitana", "Nateri", "Neddu", "Nieddu", "Ninniri", "Nonnis", 
    "Oggiano", "Onnis", "Oppus", "Orrù", "Ortu", "Pala", "Pani", 
    "Para", "Pau", "Pazza", "Perra", "Pes", "Petretto", "Piga", 
    "Piladu", "Pilia", "Pilo", "Pinna", "Pintus", "Piras", "Piredda", 
    "Piroddi", "Pisano", "Pischedda", "Piu", "Planetta", "Podda", "Porcu", 
    "Porcheddu", "Pruneddu", "Puddu", "Puggioni", "Putzolu", "Raju", "Ruiu", 
    "Scala", "Salaris", "Sanfelice", "Sanna", "Sassu", "Scano", "Schirru", 
    "Sechi", "Secci", "Serra", "Sini", "Sirigu", "Soddu", "Solinas", 
    "Spano", "Spiga", "Suella", "Tiddia", "Tidu", "Tolu", "Trudu", 
    "Tuffu", "Uras", "Usai", "Vacca", "Vargiu", "Zanda", "Zedda"
]

#   ----------------------------------------------------------------
#   |                                                              |
#   |                                                              |
#   |                        Util Functions                        |
#   |                                                              |
#   |                                                              |
#   ----------------------------------------------------------------



def split_text(text, max_chars=3000):
    """
    Return a list with the original text splitted

    :param text: The text to split
    :param max_chars: The number of characters per chunk (3000 per default)
    """
    return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]



# Source : https://stackoverflow.com/questions/55704218/how-to-check-if-pdf-is-scanned-image-or-contains-text
def get_text_percentage(file_name: str) -> float:
    """
    Calculate the percentage of document that is covered by (searchable) text.

    If the returned percentage of text is very low, the document is
    most likely a scanned PDF

    :param file_name: the file to scan
    """
    total_page_area = 0.0
    total_text_area = 0.0

    doc = fitz.open(file_name)

    for _, page in enumerate(doc):
        total_page_area = total_page_area + abs(page.rect)
        text_area = 0.0
        for b in page.get_text_blocks():
            r = fitz.Rect(b[:4])  # rectangle where block text appears
            text_area = text_area + abs(r)
        total_text_area = total_text_area + text_area
    doc.close()
    return total_text_area / total_page_area


def pseudonymize(value, category):
    """
    Pseudonymize function uses a value and a category. It will add to the mapping dictionnary
    the different entries the ner model will find to replace them in the text. It will not 
    affect the different PDF, only the data that we collected. This method allows some coherence.
    If we get multiple times the same person, she will always be categorize as 'PERSON_001'.
    However, if there is multiple person with the same name they will be considered as the same
    person.
    
    :param value: initial text to add into the mapping
    :param category: category of the data to pseudonymize
    """
    if value not in mapping[category]:
        mapping[category][value] = f"{category}{len(mapping[category]) + 1}"
    return mapping[category][value]

# Regex generated with ChatGPT
EMAIL_REGEX = r'\b[a-zA-Z0-9](?:[a-zA-Z0-9._%+\-]*[a-zA-Z0-9])?@[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9\-]*[a-zA-Z0-9])?)*\.[a-zA-Z]{2,}\b'
PHONE_REGEX = r'\+?\d[\d\s\/.-]{6,}\d'
CF_REGEX = r'\b[A-Z]{6}\d{2}[A-Z]\d{2}[A-Z]\d{3}[A-Z]\b'
PIVA_REGEX = r'\b\d{11}\b'
WEBSITE_REGEX = r"(?:https?://|www\.)[^\s]+"


def anonymize_text(text:str):
    """
    Find the data easy to recognize such as mail or phone numbers and pseudonymize them
    
    :param text: Initial text to anonymize
    """
    
    result_ner = {}

    for match in re.finditer(EMAIL_REGEX, text):
        pseudonymize(match.group(), "EMAIL")

    for match in re.finditer(PHONE_REGEX, text):
        pseudonymize(match.group(), "PHONE")

    for match in re.finditer(CF_REGEX, text):
        pseudonymize(match.group(), "ID")

    for match in re.finditer(PIVA_REGEX, text):
        pseudonymize(match.group(), "ID")

    for match in re.finditer(WEBSITE_REGEX, text):
        pseudonymize(match.group(), "ID")


    # Check if the text is long enough to make the use of nlp usefull
    if len(text) > 100:  
      
        # Search the data which can't be found with regex such as names or adress with a NER model
        # And replace them with the pseudonymized version
        doc = nlp(text)

        for ent in doc.ents:
            if ent.label_ == "PER":
                result_ner[ent.text] = "PERSON"

            elif ent.label_ in ["LOC", "GPE"]:
                result_ner[ent.text] = "ADDRESS"


        # List of persons found by the NER
        ner_persons = [ent.text for ent in doc.ents if ent.label_ == "PER"]


        for name in mostCommonNames:
            # Pseudonymize the name in the text
            if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE):
                pseudonymize(name, "PERSON")

            # Check the previous and next word if it is a name
            pattern = (
                rf"\b([A-ZÀ-Öa-zà-ö]{{2,}})\s+{re.escape(name)}\b"
                rf"|\b{re.escape(name)}\s+([A-ZÀ-Öa-zà-ö]{{2,}})\b"
            )

            # Pseudonymize every name found
            for match in re.finditer(pattern, text):
                pseudonymize(match.group(0), "PERSON")

            # Pseudonymize every name found with the context of the NER
            for ner_name in ner_persons:
                if (
                    name.lower() in ner_name.lower()
                    and ner_name.lower() != name.lower()
                ):
                    pseudonymize(ner_name, "PERSON")

    # Copy the text
    result_text=text.lower()

    # Pseudonymize every object found by the NER 
    for value, category in result_ner.items():
        pseudonymize(value, category)
                    
    # Replace the original text data's by the pseudonymize version 
    for _ , values in mapping.items():
        for original, pseudo in values.items():
            if isinstance(original, str):
                original=original.lower()

            result_text = re.sub(
                rf"\b{re.escape(original)}\b",
                pseudo,
                result_text
            )

    return result_text


def OCR_processing(img):
    """
    Preprocess the image with an OCR
    :param img: Initial image to apply the OCR
    """
    img = np.asarray(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Upscaling the image 
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Normalizing the contrast
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Denoising the image
    gray = cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)

    # Sharpening the image 
    sharpen_kernel = np.array([[0, -1,  0],
                               [-1,  5, -1],
                               [0, -1,  0]])
    gray = cv2.filter2D(gray, -1, sharpen_kernel)

    # Creating a threshold
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=21,
        C=8
    )

    # Closing small gaps inside characters
    kernel = np.ones((2, 2), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    return thresh

def fix_ocr_spacing(text):
    """
    Fix spacing between characters from scanned pdf
     
    :param text: Input text
    """
    return re.sub(r'\b(?:\w\s){2,}\w\b',lambda m: m.group().replace(" ", ""), text)



def normalized(input_text):
    """
    Return the same text without the most used unicodes and few characters not usefull
    
    :param text: Source text with unicodes
    """
    input_text = unicodedata.normalize("NFKC", input_text)
    replacements = {"\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"', "\u00ae": "",  "\u00a9": "", "\u001a": "",
            "\u001d": "", "\u0001": "", "\u0003": "", "\u0005": "", "\u0010": "", "\u0011": "", "\u0012": "", "\u0013": "", 
            "\u0014": "", "\u0015": "", "\u0016": "", "\u0017": "", "\u0018": "", "\u0019": "", "\u001c": "","\u000f": "",
            "\u001b": "","\u001d": "", "\u00b0": "", "\u2026": "...", "\u2013": "-", "\u2014": "-", "\u00bb": "", "\u00ab": "",
            "_": " ", "\n": " ", "<" : " ", ">" : " ", "\/" : "/"
        } 
    for k, v in replacements.items():
        input_text = input_text.replace(k, v)
        
    input_text = re.sub(r'[\uf000-\uf8ff]', '', input_text)

    return input_text






#   ----------------------------------------------------------------
#   |                                                              |
#   |                                                              |
#   |                             Main                             |
#   |                                                              |
#   |                                                              |
#   ----------------------------------------------------------------






with open("./classes.json") as file:
    classes = json.load(file)

reversed_classes = {}
for k,v in classes.items():
    reversed_classes[v]=k

def main():
    compteur=0
    # Get every json files in a list
    onlyfiles = [f for f in listdir("."+dir) if isfile(join("."+dir, f))]
    jsonFiles={}
    resDict={"protocolID": [], "text": [], "label": [], "encoding":[], "date": []}

    for nameFile in onlyfiles:  
        # Construct relative path for each json file
        fullName="."+dir+nameFile

        # Open the json file to get it in a dict
        with open(fullName) as file:
            f = json.load(file)

        # "jsonFiles" is dict who stock the content loaded of every json files. 
        # The key is their relative path from working dir to where they are 
        jsonFiles[fullName]=f

    # For each files in jsonFiles, getting the attachements 
    for _,v in jsonFiles.items():
        for f in v["allegati"]:
            # Checking if the attachement's absolute path is a file
            if isfile(path+dir+"/"+f):
                # Get the extension of each file
                _, fileExtension = os.path.splitext(f)
                filePath = "." + dir + f

                # If it is a pdf 
                if fileExtension==".pdf":
                    text_perc = get_text_percentage(filePath)
                    # If the pdf is scanned : apply OCR to extract text
                    if text_perc < 0.01:
                        with tempfile.TemporaryDirectory() as p:
                            images = convert_from_path(filePath, output_folder=p, dpi=150)
                            text = ""
                            for img in images:
                                upscaledImg = OCR_processing(img)
                                text += pytesseract.image_to_string(upscaledImg,lang='ita', config='--oem 1 --psm 3 -c tessedit_char_blacklist=§©®')                    

                    # Else, the file is a digital PDF : extract the text directly
                    else:
                        text = ""
                        reader = PdfReader(filePath)
                        pagesNumber = len(reader.pages)
                        for i in range(pagesNumber):
                            page = reader.pages[i]
                            text += page.extract_text()

                    # Text Preprocessing and anonymization
                    text = normalized(text.strip())
                    text = fix_text(text)
                    text = anonymize_text(text)


                    # Insert the data into a dictionnary to prepare the dataframe
                    label=v["CLASSIFICAZIONE"]
                    if(label == "243" or label == "328" or label == "329" or label == "361"):
                        label="Amministrazione generale" # Random class for the label not conventional
                        

                    resDict["protocolID"].append(v["ID"])
                    resDict["text"].append(text)
                    resDict["label"].append(label)
                    resDict["encoding"].append(reversed_classes[label]) 
                    resDict["date"].append(v["PG_DATA"])
                    compteur+=1
                    print("File number : " + str(compteur)) 
    return resDict




#   ----------------------------------------------------------------
#   |                                                              |
#   |                                                              |
#   |                    Creation of output                        |
#   |                                                              |
#   |                                                              |
#   ----------------------------------------------------------------



def output(data, map, csv):
    # Creation of the dataframe
    dataset = pd.DataFrame(data=data)
    # Creation of a CSV
    if(csv==True):
        dataset.to_csv("data.csv")

    # Creation of a JSON line
    df=dataset.to_json(orient="records", lines=True, force_ascii=False)
    with open("data.jsonl", "w", encoding="utf-8") as f:
        f.write(df)  

    with open("mapping.json", "w", encoding="utf-8") as f:
        json.dump(map, f, ensure_ascii=False, indent=4)

        
d = main()
output(d, mapping, False)