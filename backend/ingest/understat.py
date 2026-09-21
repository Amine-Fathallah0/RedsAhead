import json
import re


def extract_embedded_json(html: str, var_name: str):
    found = re.search(rf'var {var_name} = JSON\.parse\(\'(.*?)\'\);', html)      # job 1: find the scrambled text
    if found is None:
        raise KeyError(f"Variable '{var_name}' not found in page.")
    escaped = found.group(1)
    decoded = escaped.encode("latin-1", "backslashreplace").decode('unicode_escape')                     # job 2: decode the escapes
    return json.loads(decoded)                        # job 3: turn the text into Python data







