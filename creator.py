from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import os
from matplotlib import font_manager
import random

"""def bruit(image, taux):
    largeur, hauteur = image.size
    image_bruite = ImageDraw.Draw(image)
    for x in range(largeur):
        for y in range(hauteur):
            if random.random() < taux:
                gris = random.randint(0, 255)
                image_bruite.point((x, y), fill =(gris, gris, gris))
    return image"""


dossier = r"C:\pytorch\projet\trad_jpeg_png\letter_code"
if not os.path.exists(dossier):
    os.makedirs(dossier)

# Lister toutes les polices disponibles sur le système
polices_disponibles = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')

# Extraire les noms de polices (uniques) avec une taille par défaut
taille_police = 80
polices_a_exclure = ["Webdings", "Wingdings", "Wingdings 2", "Wingdings 3", "Symbol", "Segoe MDL2 Assets", 
                     "remixicon", "MS Outlook", "MT Extra", "MS Reference Specialty","Material Design Icons 5.9.55", 
                     "Material Design Icons", "HoloLens MDL2 Assets", "Font Awesome 5 Free Solid","FontAwesome", "Font Awesome 5 Brands",
                     "Font Awesome 5 Free Regular","Bookshelf Symbol 7","codicon","elusiveicons"]
noms_polices = set()
for police in polices_disponibles:
    try:
        nom = font_manager.FontProperties(fname=police).get_name()
        if not any(exclue.lower() in nom.lower() for exclue in polices_a_exclure):
             noms_polices.add((nom, police))  # Stocker le nom et le chemin
    except:
        continue

largeur, hauteur = 100, 100
couleur_fond = "white"
couleur_texte = "black"
taux_bruit = 0.1

for nom_police, chemin_police in noms_polices:
    try:
        police = ImageFont.truetype(chemin_police, taille_police)
    except:
        police = ImageFont.load_default()
        print(f"Impossible de charger la police {nom_police}, utilisation de la police par défaut.")

    for lettre in "A":#BCDEFGHIJKLMNOPQRSTUVWXYZ
        image = Image.new("RGB", (largeur, hauteur), couleur_fond)
        dessin = ImageDraw.Draw(image)
        bbox = dessin.textbbox((0, 0), lettre.upper(), font=police)
        position = ((largeur - bbox[2]) // 2, (hauteur - bbox[3]) // 2)
        dessin.text(position, lettre.upper(), fill=couleur_texte, font=police)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1)  # Ajustez la valeur pour plus ou moins de contraste
        image = image.filter(ImageFilter.GaussianBlur(radius = 0.2))
        #image = bruit(image, taux_bruit)
        chemin = os.path.join(dossier, f"{lettre.upper()}_{nom_police}.png")
        if os.path.exists(chemin):
            nom_fichier = f"{lettre.upper()}_{nom_police}.png"
            chemin = os.path.join(dossier, nom_fichier)
        image.save(chemin)

    for lettre in "a":#bcdefghijklmnopqrstuvwxyz
        image = Image.new("RGB", (largeur, hauteur), couleur_fond)
        dessin = ImageDraw.Draw(image)
        bbox = dessin.textbbox((0, 0), lettre.lower(), font=police)
        position = ((largeur - bbox[2]) // 2, (hauteur - bbox[3]) // 2)
        dessin.text(position, lettre.lower(), fill=couleur_texte, font=police)

        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1)  # Ajustez la valeur pour plus ou moins de contraste
        image = image.filter(ImageFilter.GaussianBlur(radius = 0.2))
        #image = bruit(image, taux_bruit)

        chemin = os.path.join(dossier, f"{lettre.lower()}_{nom_police}.png")
        if os.path.exists(chemin):
            nom_fichier = f"{lettre.lower()}_{nom_police}_2.png"
            chemin = os.path.join(dossier, nom_fichier)
        image.save(chemin)

print(f"Les images ont été enregistrées dans le dossier '{dossier}'.")
