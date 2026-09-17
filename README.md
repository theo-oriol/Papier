# Entraînement unique multi-ablations

Implémentation du protocole `protocole_ablations_uniques_v3.pdf` : un modèle
DINOv3-L + LoRA, agrégation MIL par attention *gated*, sept ablations tirées
en ligne sur les crops. Voir le PDF pour la spécification complète ; ce
document explique comment le code s'y branche, ce qui vient d'où, et ce qui
reste ouvert.

## Démarrer

```bash
conda activate torch   # environnement existant sur cette machine (torch 2.5.1, cuda ok)
pip install -r requirements.txt   # si besoin de compléter l'environnement

python scripts/check_dataset_integrity.py      # vérifie folds/crosswalk (§7)

# A1, A2, A7 sont déterministes (aucun paramètre tiré) — précalculées une
# fois hors ligne au lieu d'être recalculées à chaque sac qui les tire, à
# chaque époque. Nécessaire avant tout entraînement (chain.py lève une
# erreur claire si une de ces ablations est tirée et son cache n'existe pas).
python scripts/build_a1_cache.py --workers 8
python scripts/build_a2_cache.py --workers 8
python scripts/build_a7_cache.py --workers 8

python train.py --config configs/train_fold0.yaml
```

Le notebook `notebooks/01_ablation_catalogue.ipynb` montre l'effet de
chaque ablation sur un vrai spécimen (à ouvrir avec le kernel `torch`).
`scripts/benchmark_ablations.py` mesure ce qui est lent.

## Provenance des chemins (`configs/paths.yaml`)

Rien n'était fourni au départ — chaque chemin a été retrouvé sur la machine
et vérifié contre les nombres imprimés dans le protocole avant d'être
utilisé :

- **Dataset** (`NEW_Segmented-Aves-Back-NPY`) : trouvé sur le disque LaCie.
  Format vérifié directement sur un spécimen réel (`.rgb.zst` →
  `(2024, 2024, 3)` uint8 ; `.cand_*.zst` → le format exact attendu par
  `src/sampling.py`).
- **Folds** (`cross_version_1_FAM`) + **crosswalk** : trouvés dans
  `bird_project/data/`. Confirmés par coïncidence exacte avec les chiffres
  du protocole : 37890+54600+30471 images, 83+77+80 familles, 241
  familles/36 ordres/9143 espèces dans le crosswalk, et — une fois
  restreint à la vue Back présente dans ce dataset NPY — exactement
  12630+18200+10157 images de validation et 28357+22787+30830
  d'entraînement (protocole §7). `scripts/check_dataset_integrity.py`
  revérifie tout ça.
- **`color.py`** (CIELAB, colour-naming CN11, calibration Delhey) : fourni
  tel quel (copié dans `src/color.py`), avec sa ressource `w2c.mat`.
- **DINOv3-L** : poids + repo torch.hub locaux trouvés sur la machine, même
  schéma de chargement que `bird_project/src/models/backbone.py`
  (`torch.hub.load(..., source="local")`).
- **`mil5_metrics_per_class.csv`** (prévalences d'habitat, citée §1/§7) :
  introuvable sur la machine. Les poids BCE `w_c ∝ 1/f_c` sont donc
  **calculés directement depuis le fold d'entraînement**
  (`src/datasets/folds.py:class_frequencies`) plutôt que lus d'un fichier
  externe, et sauvegardés dans `manifest.json` à chaque run pour rester
  traçables. À remplacer si ce fichier est retrouvé.

## Structure

```
src/
  color.py                 fourni pour le projet — CIELAB / colour naming
  npy_io.py                 lecture/écriture des .rgb.zst
  sampling.py                positions de crop depuis les tables .cand_*.zst
  rng.py                      graines déterministes (image, condition, ...)
  chain.py                    LA chaîne canonique (protocole §2) — ordonne les 4 points d'insertion
  ablations/
    a1_gray_lstar.py ... a7_chroma_only.py    une ablation, un fichier,
      chacun 100% autonome (numpy/scipy/skimage/zstandard seulement — voir
      "Portabilité des fichiers d'ablation" ci-dessous) : copier un seul de
      ces fichiers ailleurs suffit à le réutiliser, rien d'autre du projet
      n'est nécessaire
    common.py                  Condition, groupe exclusif CIELAB
    combinations.py             tirage §4 (36 conditions)
  datasets/
    folds.py                    fold CSVs + crosswalk + poids de classe
    bag_dataset.py                Dataset d'entraînement (tirage en ligne, §4)
    eval_dataset.py                Dataset d'évaluation (conditions fixes, §7)
  model/
    backbone.py, mil.py, heads.py, bagmodel.py, losses.py
  training/
    run.py                      la boucle d'entraînement + traçabilité
  evaluation/
    conditions.py, a6_presence.py, metrics.py, run_eval.py

configs/            un fichier YAML par run ; paths.yaml/model.yaml/optim.yaml sont inclus par les autres
scripts/             build_a{1,2,7}_cache, build_a6_presence_manifest, benchmark_ablations, check_dataset_integrity, combine_fold_results,
                      build_grayscale_dataset, build_grayscale_norm_lum_dataset
notebooks/            catalogue visuel des ablations
train.py, evaluate.py  points d'entrée
```

## Portabilité des fichiers d'ablation

Chaque fichier de `src/ablations/` (A1-A7) est autonome : aucun `from ..
import` vers le reste du projet, seulement des paquets tiers (numpy, cv2,
scipy, scikit-image, zstandard selon le fichier). Copier un seul fichier
ailleurs suffit à le réutiliser.

A3, A4, A5 sont déjà minimalistes (numpy/cv2 seulement — rien à faire).
A1, A6, A7 dépendaient de `src/color.py` (~1300 lignes de CIELAB/CN11,
fourni pour le projet) ; chacun en embarque maintenant sa propre copie,
extraite programmatiquement de `color.py` via `ast` (pas de recopie à la
main — aucun risque de coquille dans du code de calibration scientifique),
et vérifiée bit-à-bit identique au résultat de `color.py` sur des données
réelles avant d'être acceptée (A2 est un cas à part, voir plus bas) :

- **A1** n'utilise que les conversions CIELAB (`rgb2lab`/`lab2rgb`) —
  quelques dizaines de lignes, pas de table externe.
- **A2 a depuis été réécrit** pour utiliser une cible calculée sur tout le
  dataset (voir "Ce qui n'a pas pu suivre le protocole à la lettre" plus
  bas) — son CIELAB n'est plus extrait de `color.py` mais recodé à la main
  (mêmes formules que `scripts/build_grayscale_norm_lum_dataset.py`),
  pour éviter une cinquième copie de la conversion sRGB↔Lab dans le projet.
  Reste néanmoins 100% autonome (numpy + zstandard seulement).
- **A7** de même, à un détail près : la fonction d'origine calcule aussi
  une répartition de la perte de gamut *par catégorie Delhey*, ce qui
  demanderait tout le classifieur CN11. Ce projet n'utilise jamais ce
  champ (`scripts/build_a7_cache.py` ne garde que les statistiques
  globales), donc la copie autonome le supprime — documenté dans le
  docstring de la fonction, vérifié pour ne rien changer aux 13 autres
  champs de métadonnées ni à l'image produite.
- **A6** ne peut pas s'alléger de la même façon : retirer une couleur
  *nommée* suppose de classifier les pixels dans les 12 catégories de
  Delhey, donc le fichier embarque le classifieur en entier — y compris
  la table de lookup CN11 (`w2c.mat`, 2,7 Mo), stockée en base64 compressé
  directement dans le `.py` plutôt que comme fichier `resources/` à côté,
  pour que ce soit vraiment un seul fichier à transférer. Résultat :
  `a6_colour_removal.py` fait ~3,7 Mo — c'est le prix de l'autonomie pour
  cette ablation-là, pas un bug.

`src/color.py` reste dans le projet : c'est la source dont ces quatre
fichiers ont été extraits, et le reste du code (`scripts/benchmark_ablations.py`,
`src/evaluation/a6_presence.py`, le notebook) continue de s'y référer pour
ses propres besoins de diagnostic — seuls les fichiers d'ablation eux-mêmes
devaient être transférables isolément.

## Jeux de données grayscale hors ligne

Deux scripts construisent une copie complète des 3 vues (Back/Belly/Side),
même mise en page que `Familly_split_no_ablation/` (un dossier à elles,
les mêmes sous-dossiers `NEW_Segmented-Aves-{view}-NPY`, `.cand_*.zst`
copiés tels quels car la géométrie ne dépend pas de la couleur) :

- **`scripts/build_grayscale_dataset.py`** — gris "A1" (CIELAB, L*
  conservé), un seul passage, embarrassingly parallel.
- **`scripts/build_grayscale_norm_lum_dataset.py`** — gris SHINE
  (Willenbockel et al. 2010, lumMatch + histMatch exact sur L*) : chaque
  spécimen est remis à l'échelle sur (M, S) de la population puis
  histogram-matché à une distribution cible commune, calculée sur
  l'ensemble des spécimens — donc trois passes (stats → cible → application),
  pas une. La cible (`target_pdf`) et M/S sont sauvegardés dans
  `runs/grayscale_norm_lum_metadata/<vue>/` et réutilisés si le script est
  relancé (`--recompute-metadata` pour forcer). Reprend la logique corrigée
  de `bird_project/scripts/generate_meta_data_luminance.py` : l'ancienne
  version (`generate_meta_data_grayscale.py`) supposait que toutes les
  images avaient le même nombre de pixels que la première, et laissait
  silencieusement de la mémoire non initialisée dans les images ayant plus
  de pixels — corrigé ici en recalculant les comptages par image (méthode
  des plus grands restes) à partir d'une distribution cible normalisée.
  Vérifié après coup : l'histogramme de chaque image standardisée colle à
  moins de 0,0002 de la cible.

## Traçabilité d'un run

`python train.py --config configs/train_fold0.yaml` crée
`runs/main_fold0/` contenant :

- `config.yaml` — la config résolue (tous les `includes` fusionnés), donc
  le run est reproductible à partir de ce seul fichier ;
- `manifest.json` — horodatage, commit git (si dépôt), versions
  python/torch, GPU, poids de classe BCE calculés ;
- `metrics.csv` — une ligne par époque (loss, KL, BCE, lr, durée) ;
- `checkpoints/{last,best}.pt` — `best` = plus faible loss d'entraînement
  lissée (le seul critère utilisé pour l'arrêt anticipé, §1 — aucune
  métrique de validation n'entre dans ce choix, conformément au protocole).

## Ce qui n'a pas pu suivre le protocole à la lettre

- **A2 n'utilise plus la cible fixe (50, 15) du protocole.** §3 spécifie
  `L' = (L-m)*(15/s)+50` avec `m, s` propres à chaque spécimen mais une
  cible *fixe*, identique pour tout le monde. Sur demande explicite,
  `src/ablations/a2_gray_standardized.py` calcule maintenant cette cible —
  moyenne, écart-type, **et histogramme complet** — sur l'ensemble du
  dataset (lumMatch + histMatch exact, Willenbockel et al. 2010, la
  méthode SHINE déjà utilisée par `build_grayscale_norm_lum_dataset.py`
  pour son propre jeu de données grayscale séparé). Concrètement : chaque
  spécimen est maintenant appariché à la distribution *moyenne* de la
  population plutôt que simplement recentré sur une cible fixe — un
  changement de méthode, pas juste de paramètres. `scripts/build_a2_cache.py`
  fait donc trois passes (stats → cible → application) au lieu d'une, et
  sauvegarde M/S/`target_pdf` dans `runs/a2_metadata/` pour ne pas repayer
  les deux premières passes à chaque relance (`--recompute-metadata` pour
  forcer). Vérifié sur données réelles : l'histogramme d'un spécimen
  standardisé colle à moins de 0,0002 de la cible. Toute cache A2
  construite avant ce changement est invalide et doit être reconstruite.
- **`batch_size: 64` (1024 crops/pas) ne tient pas sur ce GPU.** Le GPU de
  cette machine (Quadro RTX 5000, 16 Go) sature déjà à ~5.7 Go pour un
  *seul* sac (16 crops) en bf16 — mesuré directement, voir
  `configs/optim.yaml`. Le `Trainer` fait donc de l'accumulation de
  gradient (`micro_batch_size: 1` par défaut) pour retrouver le batch
  effectif de 64 sans jamais charger plus d'un sac à la fois sur le GPU.
  À augmenter si vous avez un GPU plus gros.
- **A1/A6/A7 étaient très lents — A1 et A7 corrigés, A6 partiellement.**
  Le benchmark initial donnait ~1,7s/image pour A1, ~31s pour A6, ~17s pour
  A7, dominés par `color.classify_delhey_rgb` (ou son équivalent
  `rgb2lab`/`lab2rgb` pour A1). Diagnostic (`cProfile` sur un vrai
  spécimen) : (1) tout tournait sur le canevas 2024×2024 entier alors que
  seule la fraction "oiseau" du masque compte ; (2) A6 classifiait l'image
  deux fois (présence, puis ablation) ; (3) **A1 et A7 sont des fonctions
  pures du spécimen** (aucun paramètre tiré), donc les recalculer à chaque
  sac qui les tire, à chaque époque, était du travail refait pour rien.
  Fixes appliqués :
    - `src/color.py` : `_classify_delhey_rgb_details` restreint tous ses
      calculs à la boîte englobante du masque (résultat identique bit à
      bit sur les pixels du masque, vérifié directement) et `_sigmoid` est
      devenu branchless (`exp(-logaddexp(0,-x))` au lieu d'un masquage
      booléen) ;
    - `src/ablations/a6_colour_removal.py:draw_and_apply` classifie une
      seule fois et réutilise ce résultat pour choisir *et* retirer la
      couleur (`color.ablate_delhey_colour_from_details`) ;
    - **A1 et A7 sont maintenant précalculées hors ligne**
      (`scripts/build_a1_cache.py`, `scripts/build_a7_cache.py`, même
      principe qu'A2) et lues comme A2 (`chain.py` step 0) au lieu d'être
      recalculées en ligne. A7 perdait ses statistiques de gamut (utile
      pour QC) en devenant un cache — elles sont maintenant écrites une
      fois pour toutes dans `runs/a7_gamut_metadata.csv` par le script de
      build plutôt que recalculées à chaque tirage.
  Résultat mesuré (`scripts/benchmark_ablations.py` / notebook, mêmes
  spécimens) : **A1 1,7s → 0,3s, A7 17s → 0,3s (cache), A6 31s → 7,3s
  (×4,2, reste en ligne)**. A6 ne peut pas être mis en cache de la même
  façon : sa catégorie est tirée *par sac*, donc il faudrait cacher la
  classification (les labels) plutôt que le résultat final — pas encore
  fait, prochaine étape si A6 reste le facteur limitant. Le reste du coût
  d'A6 (`classify_delhey_rgb` seul ~4-5s/image) est la classification
  CIELAB/CN11 elle-même — pour aller plus loin il faudrait retravailler
  cette logique de calibration en profondeur (float32, vectorisation plus
  fine, GPU), plus invasif dans un fichier fourni pour le projet.
  `scripts/build_a6_presence_manifest.py` (nécessaire pour la grille
  d'évaluation A6, §7) reste donc long à faire tourner une première fois
  (~6h à 8 workers pour les ~41 000 spécimens, contre ~17h avant les fixes)
  mais reprenable.
- **A6 en évaluation reste "en discussion" dans le protocole lui-même**
  (§7) : ce code implémente la définition actuelle (une passe par
  catégorie chromatique présente), à ajuster si cette discussion aboutit
  ailleurs.
- **Checkpoints non compactés** : `Trainer._save_checkpoint` sauvegarde le
  backbone entier (poids gelés + LoRA) à chaque epoch pour rester
  autonome/simple à recharger, plutôt que juste les poids LoRA — plus
  simple, plus gros sur disque.
