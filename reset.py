import json

notebook_path = 'notebook/notebook.ipynb'

with open(notebook_path, 'rt') as file_in:
    notebook_data = json.load(file_in)

execution_counter = 1

for cell in notebook_data['cells']:
    if 'execution_count' not in cell:
        continue

    cell['execution_count'] = execution_counter

    for output in cell.get('outputs', []):
        if 'execution_count' in output:
            output['execution_count'] = execution_counter

    execution_counter += 1

with open(notebook_path, 'wt') as file_out:
    json.dump(notebook_data, file_out, indent=1)
    
print('done')