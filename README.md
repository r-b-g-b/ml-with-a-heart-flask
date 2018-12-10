This code implements a simple API for the model descibred in [ml-with-a-heart](https://github.com/r-b-g-b/ml-with-a-heart). Email me for the address. Here is an example showing how to retrieve a prediction for a set of factors:

```python
import json
from pprint import pprint
import numpy as np
import requests

host = 'apple-orange-pear-banana.herokuapp.com'
url = f'http://{host}/predict'

data = [{'patient_id': 'patient-0',
         'slope_of_peak_exercise_st_segment': 1,
         'thal': 'normal',
         'resting_blood_pressure': 128,
         'chest_pain_type': 2,
         'num_major_vessels': 0,
         'fasting_blood_sugar_gt_120_mg_per_dl': 0,
         'resting_ekg_results': 2,
         'serum_cholesterol_mg_per_dl': 308,
         'oldpeak_eq_st_depression': 0.0,
         'sex': 1,
         'age': 45,
         'max_heart_rate_achieved': 170,
         'exercise_induced_angina': 0
         },
        {'patient_id': 'patient-0',
         'slope_of_peak_exercise_st_segment': 1,
         'thal': 'reversible_defect',
         'resting_blood_pressure': 128,
         'chest_pain_type': 4,
         'num_major_vessels': 3,
         'fasting_blood_sugar_gt_120_mg_per_dl': 0,
         'resting_ekg_results': 2,
         'serum_cholesterol_mg_per_dl': 300,
         'oldpeak_eq_st_depression': 0.0,
         'sex': 0,
         'age': 60,
         'max_heart_rate_achieved': 170,
         'exercise_induced_angina': 0
         }]

r = requests.post(url, json=json.dumps(data))

pprint(json.loads(r.text))

{'prediction': [0, 1],
 'probability': [[0.4180, 0.6418]]}
```