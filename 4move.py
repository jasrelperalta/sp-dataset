# Move all movie posters from different folders inside 'posters2' to a single folder named 'xposters'

import os
import shutil

# create a new folder named 'xposters' to store all movie posters
if not os.path.exists('xposters'):
    os.makedirs('xposters')

# move all movie posters from different folders inside 'posters2' to a single folder named 'xposters'
for root, dirs, files in os.walk('posters2'):
    for file in files:
        shutil.move(os.path.join(root, file), 'xposters')

print("Movie posters moved successfully")
