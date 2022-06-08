import pikepdf
import os

for filename in os.listdir('NMIMS copy'):
    if filename.__contains__('.pdf'):
        pdf = pikepdf.open(r"NMIMS copy/" + str(filename))
        # pdf = pikepdf.open(r"NMIMS copy/" + str(filename), password = '<password here>')
        pdf.save('Extracted/' + str(filename))
