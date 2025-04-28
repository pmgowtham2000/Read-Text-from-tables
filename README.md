# Read-Text-from-tables
The above code uses following approach to read text from tables and their respective images:
1. Yolo model is trained on identifying columns
2. Using trained model, the columns are extracted from tables
3. Since needed text and images occur on same row, bounding boxes with needed x1 and x2 coordinates are extracted
4. Now, easyOCR acts on the text to read it

# Install the dependencies
     pip install -r requirements.txt
