import pandas as pd

data = {'Parameter': [],
        'Value': [],
        'Unit': []}

data['Parameter'].extend(['DUGR_I', 'k^2_I', 'A_p_new_I', 'I'])
data['Value'].extend([1, 2, 3, 4])
data['Unit'].extend(['None', 'None', 'mm^2', 'cd'])

df = pd.DataFrame(data)

result = df.to_json()
print(str(result))