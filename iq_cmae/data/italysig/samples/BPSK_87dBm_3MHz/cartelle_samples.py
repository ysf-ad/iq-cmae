import os

for i in range(1, 21):
    folder_name = f"sample{i}"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    else:
        print(f"La cartella {folder_name} esiste già.")
