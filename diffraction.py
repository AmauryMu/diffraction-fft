"""
Figures de diffraction de Fraunhofer par FFT implémentée à la main.

On simule la diffraction d'une onde plane par trois ouvertures 2D
(trou carré, trou circulaire, double trou d'Young). En champ lointain,
l'amplitude diffractée est proportionnelle à la transformée de Fourier
de l'ouverture : on la calcule avec une FFT récursive (Cooley-Tukey,
lemme de Danielson-Lanczos) codée sans numpy.fft.

Projet M1 Physique, CY Cergy Paris Université (janvier 2026).

Utilisation :
    python diffraction.py               # affiche les figures
    python diffraction.py --animation   # anime l'effet de la taille des ouvertures (environ 3 min)
    python diffraction.py --sauver      # enregistre la figure et l'animation dans figures/
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

SAUVER = "--sauver" in sys.argv
ANIMATION = "--animation" in sys.argv


# ---------------------------------------------------------------------------
# FFT implémentée à la main
# ---------------------------------------------------------------------------

def fft_1d(x):
    """
    Implémentation récursive de la FFT (Cooley-Tukey).
    Basée sur le lemme de Danielson-Lanczos.
    x : vecteur d'entrée (sa taille doit être une puissance de 2)
    """
    N = len(x)

    # Cas de base : la transformée d'un seul point est le point lui-même
    if N <= 1:
        return x

    # Division : on sépare les indices pairs (f_2j) et impairs (f_2j+1)
    even = fft_1d(x[0::2])
    odd = fft_1d(x[1::2])

    # Facteur "twiddle" : u = exp(-2i * pi * k / N)
    # Signe moins : convention de la DFT directe
    k = np.arange(N // 2)
    u = np.exp(-2j * np.pi * k / N)

    # Combinaison (papillon) :
    # F_n       = F_pair + u * F_impair
    # F_(n+N/2) = F_pair - u * F_impair
    terme_impair = u * odd
    return np.concatenate([even + terme_impair, even - terme_impair])


def fft_2d(image):
    """
    FFT 2D par séparabilité : FFT 1D sur les lignes, puis sur les colonnes.
    """
    # FFT 1D sur chaque ligne
    lignes = np.array([fft_1d(ligne) for ligne in image])

    # On transpose pour traiter les colonnes comme des lignes
    colonnes = lignes.T

    # FFT 1D sur chaque colonne
    resultat_transpose = np.array([fft_1d(col) for col in colonnes])

    # On retranspose pour retrouver l'orientation d'origine
    return resultat_transpose.T


def calcul_diffraction(image):
    """
    Intensité diffractée I = |TF(ouverture)|^2, recentrée sur l'ordre zéro.
    """
    tf = fft_2d(image)

    # fftshift ne fait que réarranger les indices pour placer la fréquence
    # nulle au centre de l'image (aucun calcul de Fourier ici)
    tf_centree = np.fft.fftshift(tf)

    # Aucune normalisation (certaines conventions divisent la DFT par N) :
    # seule la forme de la figure nous intéresse (I proportionnelle à |A|^2)
    return np.abs(tf_centree) ** 2


# ---------------------------------------------------------------------------
# Construction des ouvertures
# ---------------------------------------------------------------------------

N = 512  # Taille de la grille : puissance de 2 obligatoire (ici 2^9)
centre = N // 2
y, x = np.ogrid[:N, :N]  # Grille de coordonnées pour les équations de cercles

# 1. Trou carré (côté 2 * demi = 30 pixels)
image_carre = np.zeros((N, N))
demi = 15
image_carre[centre - demi:centre + demi, centre - demi:centre + demi] = 1
intensite_carre = calcul_diffraction(image_carre)

# 2. Trou circulaire (rayon 15 pixels)
image_cercle = np.zeros((N, N))
rayon = 15
masque = (x - centre) ** 2 + (y - centre) ** 2 <= rayon ** 2
image_cercle[masque] = 1
intensite_cercle = calcul_diffraction(image_cercle)

# 3. Double trou (trous d'Young) : deux cercles décalés de ±ecart
image_double = np.zeros((N, N))
r_double = 15  # rayon de chaque trou
ecart = 30     # distance au centre (distance entre les trous = 2 * ecart)
m1 = (x - (centre - ecart)) ** 2 + (y - centre) ** 2 <= r_double ** 2
m2 = (x - (centre + ecart)) ** 2 + (y - centre) ** 2 <= r_double ** 2
image_double[m1] = 1
image_double[m2] = 1
intensite_double = calcul_diffraction(image_double)


# ---------------------------------------------------------------------------
# Vérifications
# ---------------------------------------------------------------------------

# 1. Notre FFT doit donner le même résultat que numpy.fft.fft2
ref = np.fft.fft2(image_cercle)
ecart_relatif = np.max(np.abs(fft_2d(image_cercle) - ref)) / np.max(np.abs(ref))
print(f"Écart relatif max entre notre FFT et numpy.fft.fft2 : {ecart_relatif:.1e}")

# 2. Trou carré : premier minimum de sinc^2 attendu à N / a pixels du centre
a_px = 2 * demi
coupe = intensite_carre[centre, centre:]
m_mesure = next(m for m in range(1, len(coupe) - 1)
                if coupe[m] < coupe[m - 1] and coupe[m] <= coupe[m + 1])
print(f"Trou carré, 1er minimum : mesuré à {m_mesure} px, "
      f"théorie N/a = {N / a_px:.2f} px")


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

cas = [
    (image_carre, intensite_carre, "Ouverture : carré", "Diffraction : croix (sinc²)"),
    (image_cercle, intensite_cercle, "Ouverture : cercle", "Diffraction : tache d'Airy"),
    (image_double, intensite_double, "Ouverture : double trou", "Diffraction : franges d'Young"),
]

plt.figure(figsize=(12, 12))
for ligne, (ouverture, intensite, titre_ouv, titre_diff) in enumerate(cas):
    plt.subplot(3, 2, 2 * ligne + 1)
    plt.imshow(ouverture, cmap="afmhot")
    plt.title(titre_ouv)
    plt.axis("off")

    plt.subplot(3, 2, 2 * ligne + 2)
    # Échelle log(I + 1) pour voir les maxima secondaires, beaucoup moins
    # intenses que le pic central (le +1 évite log(0))
    plt.imshow(np.log(intensite + 1), cmap="afmhot")
    plt.title(titre_diff)
    plt.colorbar(label="Intensité (log)")
    plt.axis("off")

plt.tight_layout()

if SAUVER:
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/diffraction.png", dpi=90)
    print("Figure enregistrée : figures/diffraction.png")
elif not ANIMATION:
    plt.show()


# ---------------------------------------------------------------------------
# Animation : effet de la taille de l'ouverture et de l'écart entre les trous
# ---------------------------------------------------------------------------

def animation_ouvertures(fichier=None):
    """
    Partie 1 : un trou circulaire dont le rayon grandit de 4 à 32 pixels.
    Partie 2 : deux trous de rayon 8 dont l'écart grandit de 30 à 120 pixels.
    Le cercle en pointillés marque le 1er anneau noir d'Airy prévu par la
    théorie (rayon 1,22 N / D) ; le titre donne l'interfrange prévu (N / d).
    """
    cas = [("cercle", r) for r in range(4, 33, 2)] + [("double", e) for e in range(15, 61, 3)]
    zoom_ouv, zoom_diff = 80, 128  # demi-largeur des zones affichées (pixels)

    fig, (ax_ouv, ax_diff) = plt.subplots(1, 2, figsize=(11, 5.6))
    img_ouv = ax_ouv.imshow(np.zeros((2 * zoom_ouv, 2 * zoom_ouv)), cmap="afmhot", vmin=0, vmax=1)
    img_diff = ax_diff.imshow(np.zeros((2 * zoom_diff, 2 * zoom_diff)), cmap="afmhot", vmin=-6, vmax=0)
    anneau = plt.Circle((zoom_diff, zoom_diff), 1, fill=False, ls="--", color="cyan", lw=1)
    ax_diff.add_patch(anneau)
    for ax in (ax_ouv, ax_diff):
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(img_diff, ax=ax_diff, label="log10(I / I max)")

    def image(k):
        forme, valeur = cas[k]
        ouverture = np.zeros((N, N))
        if forme == "cercle":
            ouverture[(x - centre) ** 2 + (y - centre) ** 2 <= valeur ** 2] = 1
            D = 2 * valeur
            anneau.set_radius(1.22 * N / D)
            anneau.set_visible(True)
            ax_ouv.set_title(f"Trou circulaire\ndiamètre D = {D} px")
            ax_diff.set_title(f"Tache d'Airy\n1er anneau noir prévu (pointillés) : 1.22 N/D = {1.22 * N / D:.1f} px")
        else:
            r = 8
            ouverture[(x - (centre - valeur)) ** 2 + (y - centre) ** 2 <= r ** 2] = 1
            ouverture[(x - (centre + valeur)) ** 2 + (y - centre) ** 2 <= r ** 2] = 1
            d = 2 * valeur
            anneau.set_visible(False)
            ax_ouv.set_title(f"Deux trous\nséparés de d = {d} px")
            ax_diff.set_title(f"Franges d'Young\ninterfrange prévu : N/d = {N / d:.1f} px")

        intensite = calcul_diffraction(ouverture)
        intensite = np.log10(intensite / intensite.max() + 1e-12)
        img_ouv.set_data(ouverture[centre - zoom_ouv:centre + zoom_ouv, centre - zoom_ouv:centre + zoom_ouv])
        img_diff.set_data(intensite[centre - zoom_diff:centre + zoom_diff, centre - zoom_diff:centre + zoom_diff])
        return img_ouv, img_diff, anneau

    fig.tight_layout()
    fig.subplots_adjust(top=0.85)  # place pour les titres sur deux lignes
    anim = FuncAnimation(fig, image, frames=len(cas), interval=500, blit=False)
    if fichier:
        anim.save(fichier, writer=PillowWriter(fps=2), dpi=70)
        plt.close(fig)
        print(f"Animation enregistrée : {fichier}")
    else:
        plt.show()


if SAUVER:
    animation_ouvertures("figures/tailles_ouvertures.gif")
elif ANIMATION:
    animation_ouvertures()
