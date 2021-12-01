- **i.** Tesseract isn’t very good at multi-column OCR, especially if your image is noisy
- **ii.** You may need first to detect the table in the image before you can OCR it
- **iii.** Your OCR engine (Tesseract, cloud-based, etc.) may correctly OCR the text but be
unable to associate the text into columns/rows
  

**Project directory structure**:
```
|-- michael_jordan_stats.png
|-- multi_column_ocr.py
|-- results.csv
```

```
python multi_column_ocr.py --image michael_jordan_stats.png --output results.csv
```