import re

group9 = []
member_names = ["kristo", "Dzaki", "Ferdiansyah", "Marselina", "Nuryanti", "zalfa", "adelia"]

for name in member_names:
    pattern = r"o"
    if re.search(pattern, name):
        group9.append(name.upper())
    else:
        group9.append(name)
    
print(group9)
