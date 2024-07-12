'''
At the script level, we are using pylint to disable 
duplicate-code warnings
'''

# pylint: disable=duplicate-code

import json

import requests
from deepdiff import DeepDiff

with open('event.json', 'rt', encoding='utf-8') as f_in:
    event = json.load(f_in)


url = 'http://localhost:8080/2015-03-31/functions/function/invocations'
actual_response = requests.post(url, json=event).json()
print('actual response:')

print(json.dumps(actual_response, indent=2))

expected_response = {
    'predictions': [
        {
            'model': 'ride_duration_prediction_model',
            'version': 'Test123', # None 
            'prediction': {
                'ride_duration': 21.3,
                'ride_id': 256,
            },
        }
    ]
}


diff = DeepDiff(actual_response, expected_response, significant_digits=1)
print(f'diff={diff}')

# "Type changes" happen when there is no value ( equal to "None" ) specified 
# in one of the keys and that key is compared with the same key in the 
# second dictionary using DeepDiff package
assert 'type_changes' not in diff
# "Value changed" happen when the numerical value does not exactly match 
#  between the 2 dictionaries. Setting the significant digits will 
# compare numbers only to the desired precision
assert 'values_changed' not in diff
