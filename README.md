This code implements a simple API for the model descibred in [ml-with-a-heart](https://github.com/r-b-g-b/ml-with-a-heart). It is running at http://gentle-brushlands-69278.herokuapp.com/predict. Feel free to test it out by sending POST requests of form:

```python
import json
import requests
import numpy as np


host = 'gentle-brushlands-69278.herokuapp.com'

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
         'exercise_induced_angina': 0},
         # ... dicts for other samples here ...
         ]

r = requests.post(url, json=json.dumps(data))

print(json.loads(r.text))
```
