import easyocr
from PIL import Image
import numpy as np

# Initialize the reader once (this downloads the model on the first run)
# 'bn' stands for Bengali
reader = easyocr.Reader(['bn'], gpu=True) # Set gpu=False if you don't have a CUDA GPU

def ocr_trocr(image_path):
    """
    Using EasyOCR as the baseline for Bengali Scene Text.
    Returns: (predicted_text, confidence_score)
    """
    # EasyOCR works best with OpenCV/Numpy arrays or file paths
    results = reader.readtext(image_path)
    
    if not results:
        return "", 0.0
    
    # results format: [([[box]], "text", confidence), ...]
    # We combine all detected text for the baseline
    pred_text = " ".join([res[1] for res in results])
    confidence = np.mean([res[2] for res in results]) if results else 0.0
    
    return pred_text, confidence