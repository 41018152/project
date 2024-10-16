from openalpr import Alpr

alpr = Alpr('us', 'path_to_openalpr.conf', 'path_to_runtime_data')
if not alpr.is_loaded():
    print("Error loading OpenALPR")
else:
    print("OpenALPR loaded successfully")
