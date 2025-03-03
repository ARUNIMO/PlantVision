import pandas as pd
from googletrans import Translator

# Initialize the Translator
translator = Translator()

# Function to translate a text to Tamil
def translate_text(text, dest_lang="ta"):
    if pd.isna(text) or text.strip() == "":
        return text  # Skip empty cells
    try:
        return translator.translate(text, dest=dest_lang).text
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # Return original if translation fails

# Translate disease_info.csv
disease_info = pd.read_csv("disease_info.csv")

for column in ["disease_name", "description", "Possible Steps"]:
    disease_info[column + "_ta"] = disease_info[column].apply(lambda x: translate_text(str(x)))

disease_info.to_csv("disease_info_tamil.csv", index=False, encoding="utf-8")

# Translate supplements.csv
supplement_info = pd.read_csv("supplements.csv")

for column in ["supplement name"]:
    supplement_info[column + "_ta"] = supplement_info[column].apply(lambda x: translate_text(str(x)))

supplement_info.to_csv("supplements_tamil.csv", index=False, encoding="utf-8")

print("Translation completed! Tamil CSV files saved.")
