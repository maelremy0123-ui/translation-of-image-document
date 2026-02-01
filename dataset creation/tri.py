import os
import shutil
from sklearn.model_selection import train_test_split

dossier_init = r"C:\pytorch\projet\trad_jpeg_png\letter_code"
dossier_final = r"C:\pytorch\projet\trad_jpeg_png\letter_code\dataset_2"

# Créer les dossiers pour les majuscules et minuscules dans train et test
for path in ["train", "test"]:
    os.makedirs(os.path.join(dossier_final, path, "majuscule"), exist_ok=True)
    for lettre in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        os.makedirs(os.path.join(dossier_final, path, "majuscule", lettre), exist_ok=True)

    os.makedirs(os.path.join(dossier_final, path, "minuscule"), exist_ok=True)
    for lettre in "abcdefghijklmnopqrstuvwxyz":
        os.makedirs(os.path.join(dossier_final, path, "minuscule", lettre), exist_ok=True)

# Traiter les majuscules
for lettre in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    img = [f for f in os.listdir(dossier_init) if f.endswith(".png") and f.startswith(lettre)]
    print(f"\nOn est à la lettre majuscule {lettre}.")
    print(f"Images trouvées : {img}")

    if len(img) < 2:
        print(f"Skipping {lettre}: seulement {len(img)} image(s) trouvée(s).")
        continue

    try:
        train_img, test_img = train_test_split(img, test_size=0.2, random_state=100)
        for img_file in train_img:
            shutil.copy(
                os.path.join(dossier_init, img_file),
                os.path.join(dossier_final, "train", "majuscule", lettre, img_file)
            )
        for img_file in test_img:
            shutil.copy(
                os.path.join(dossier_init, img_file),
                os.path.join(dossier_final, "test", "majuscule", lettre, img_file)
            )
    except Exception as e:
        print(f"Erreur pour la lettre {lettre} : {e}")

    for f in img:
        os.remove(os.path.join(dossier_init, f))

# Traiter les minuscules
for lettre in "abcdefghijklmnopqrstuvwxyz":
    img = [f for f in os.listdir(dossier_init) if f.endswith(".png") and f.startswith(lettre)]
    print(f"\nOn est à la lettre minuscule {lettre}.")
    print(f"Images trouvées : {img}")

    if len(img) < 2:
        print(f"Skipping {lettre}: seulement {len(img)} image(s) trouvée(s).")
        continue

    try:
        train_img, test_img = train_test_split(img, test_size=0.2, random_state=100)
        for img_file in train_img:
            shutil.copy(
                os.path.join(dossier_init, img_file),
                os.path.join(dossier_final, "train", "minuscule", lettre, img_file)
            )
        for img_file in test_img:
            shutil.copy(
                os.path.join(dossier_init, img_file),
                os.path.join(dossier_final, "test", "minuscule", lettre, img_file)
            )
    except Exception as e:
        print(f"Erreur pour la lettre {lettre} : {e}")

    for f in img:
        os.remove(os.path.join(dossier_init, f))

print("Dataset OK : les images ont été organisées par majuscules et minuscules.")
