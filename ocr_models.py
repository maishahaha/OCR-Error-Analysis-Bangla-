import easyocr
from PIL import Image
import numpy as np
import pytesseract
import torch 

use_gpu = False
reader_easy = easyocr.Reader(['bn'], gpu=use_gpu)

def ocr_easyocr(image_path):
    """Predict text and confidence using EasyOCR"""
    try:
        results = reader_easy.readtext(image_path)
        if not results:
            return "", 0.0
        
        # Combine text and average the confidence
        pred_text = " ".join([res[1] for res in results])
        confidence = np.mean([res[2] for res in results])
        return pred_text.strip(), confidence
    except Exception as e:
        print(f"EasyOCR Error on {image_path}: {e}")
        return "", 0.0

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_tesseract(image_path):
    """
    Returns (text, average_confidence)
    Confidence is normalized to 0.0 - 1.0
    """
    try:
        # We use image_to_data to get 'conf' values
        data = pytesseract.image_to_data(Image.open(image_path), lang='ben', output_type=pytesseract.Output.DICT)
        
        # Filter out confidence -1 (noise/blocks) and empty strings
        conf_scores = [float(conf) for i, conf in enumerate(data['conf']) if data['text'][i].strip() != "" and conf != "-1"]
        text_list = [text for text in data['text'] if text.strip() != ""]
        
        full_text = " ".join(text_list).strip()
        avg_conf = (sum(conf_scores) / len(conf_scores)) / 100.0 if conf_scores else 0.0
        
        return full_text, avg_conf
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return "", 0.0



# --- Bongo ---