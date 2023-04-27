# Performance-Metrics-For-POS-And-Chunk
Performance-Metrics-For-POS-And-Chunk
# Performance Report Generation for POS Tagging
python precision_recall_score_pos_tagging.py --gold sample_gold_pos.txt --pred sample_pred_pos.txt --output sample_class_report_pos.txt
<br>
Arguments:
<br>
gold: Contains the gold pos tags in conll format (each line contains the gold pos tag, sentences are separated by a blank line)
<br>
pred: Contains the predicted pos tags in conll format (each line contains the gold pos tag, sentences are separated by a blank line)
<br>
Output: Output will be generated with precision, recall, F1-scores POS tag wise

# Performance Report Generation for Chunking
python precision_recall_score_chunking.py --input sample_chunk_data.txt --output sample_class_report_chunks.txt
<br>
Arguments:
<br>
input: The input files will contain 4 columns separated by tabs (conll format) where the 1st column is the token, 2nd column is the POS tag, 3rd column is the gold chunk tag, and 4th column is the predicted chunk tag
<br>
output: Output will be generated with precision, recall, F1-scores chunk tag wise

# To install the requirred packages, do the following
## pip install -r requirements.txt
