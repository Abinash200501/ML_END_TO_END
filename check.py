# Run this in a Python shell or script in your project root

import os

def find_bad_file(root='.'):
    for subdir, _, files in os.walk(root):
        for file in files:
            filepath = os.path.join(subdir, file)
            if not file.endswith('.py'):
                continue
            try:
                with open(filepath, encoding='utf-8') as f:
                    f.read()
            except UnicodeDecodeError as e:
                print(f"Problematic file: {filepath}")
                print(e)

find_bad_file()
