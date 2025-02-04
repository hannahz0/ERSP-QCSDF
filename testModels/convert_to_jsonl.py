# Helper Script to convert tsv (Tab-Separated Values) into jsonl files

# usage: 
# python convert_to_jsonl.py input.tsv output.jsonl "id,text,category"

import csv
import json
import sys

def tsv_to_jsonl(tsv_file, jsonl_file, headers):
    with open(tsv_file, 'r', encoding='utf-8') as tsv_f, open(jsonl_file, 'w', encoding='utf-8') as jsonl_f:
        reader = csv.DictReader(tsv_f, delimiter='\t', fieldnames=headers)  # Read as TSV with custom headers
        next(reader)  # Skip the header row
        for row in reader:
            json_line = json.dumps(row, ensure_ascii=False)  # Convert row to JSON
            jsonl_f.write(json_line + '\n')  # Write JSON line

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py input.tsv output.jsonl \"id,text\"")
    else:
        headers = sys.argv[3].split(',')
        tsv_to_jsonl(sys.argv[1], sys.argv[2], headers)
