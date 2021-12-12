_author_ = 'compiler'

# Create input and output folders before running

import pikepdf
import os

for filename in os.listdir('Input'):
    if filename.__contains__('.pdf'):
        pdf = pikepdf.open(r"Input/" + str(filename), password='<passwordHere>')
        pdf.save('Output/' + str(filename))