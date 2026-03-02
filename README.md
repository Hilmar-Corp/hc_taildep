# hc_taildep : Dépendance de queue et évaluation de copules régimes-aware (BTC/ETH)

## Objet

Ce dépôt implémente une chaîne de recherche orientée **preuves** (artefacts) pour mesurer et tester la **dépendance de queue** (*tail dependence*) sur des actifs numériques (BTC/ETH en priorité), et pour évaluer la stabilité de ces mesures sous différents régimes (**calm / stress**, et variantes). L’objectif n’est pas la “prédiction de prix”, mais la production de résultats **opposables, auditables et reconstructibles**, utilisables comme composant de **gouvernance du risque** (diagnostics de stabilité, tests de puissance, fallbacks, et preuves associées).

La philosophie est **contractuelle** : à partir de **données** et de **configurations** explicites, la pipeline doit reconstruire de façon déterministe les **tables**, **figures** et **manifests** décrits dans le papier. Les sorties sont conçues pour supporter une revue externe (traçabilité, reproductibilité, absence d’ajustements manuels *ex post*).

## Périmètre (résumé)

Le dépôt couvre :

- La mesure de dépendance de queue sous différentes familles de copules (*t-copula* et variantes ; copules asymétriques pour documenter l’asymétrie **λ_L / λ_U**).
- L’analyse par régimes (**calm / stress**), y compris des contrôles de puissance et de comparabilité.
- Des comparaisons de politiques / modèles (**static / threshold / logistic gating / markov switching**), et des diagnostics d’impact via **VaR/ES**.
- Des tests de calibration (**exceedance**, **Kupiec POF**) et des éléments de reproductibilité (**provenance**, **manifest SHA-256**).

Les livrables prioritaires sont des artefacts (**CSV/PNG/JSON**) ; chaque bloc du papier renvoie à des preuves matérialisées, produites par la pipeline.

## Organisation du dépôt

- `src/hc_taildep/` : code applicatif, logique de calcul (copules, régimes, scoring, impact VaR/ES, tests).
- `paper/` : génération des artefacts “papier” (tables/figures/manifests), plus spécification de build.
- `data/processed/` : intrants et sorties intermédiaires de recherche (potentiellement volumineux). Par défaut, ces données ne sont pas versionnées dans Git.
- `paper/out/<paper_id>/` : sortie “camera-ready” (tables, figures, manifest) utilisée pour revue et diffusion.

## Reproductibilité et auditabilité

Chaque exécution “paper builder” produit un répertoire `paper/out/<paper_id>/` contenant :

- `tables/` : tables stabilisées (tri déterministe, arrondis contrôlés).
- `figures/` : figures rendues depuis les tables/CSV (backend non-interactif).
- `manifest.json` : provenance et empreintes SHA-256 des intrants/sortants.

Définition contractuelle des empreintes :

- `h(f) = sha256(bytes(f))`

Les artefacts (tables/figures) sont conçus pour être reconstruits depuis : **configs + provenance + hashes**, sans intervention manuelle.

## Environnement d’exécution

Le dépôt est conçu pour être exécuté dans un environnement Python isolé (**venv**). Les figures sont rendues en backend non-interactif (**Matplotlib Agg**) afin d’assurer une exécution stable en CI ou sur serveur.

## Installation

Environnement Python isolé recommandé :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```
## Exécution “one-shot” 

Le “one-shot” de reconstruction du pack camera-ready (tables, figures, manifest) est défini dans paper/LOCK.md et s’exécute via la spécification paper/paper_spec.yaml.

Commande :

```bash
python -m paper.make_paper --spec paper/paper_spec.yaml
```

Cette commande :
	•	lit la spécification paper/paper_spec.yaml (identifiant de pack, chemins des runs sources, exigences minimales),
	•	reconstruit les artefacts dans paper/out/<paper_id>/,
	•	écrit paper/out/<paper_id>/manifest.json (provenance + SHA-256).

Références internes :
	•	Spécification : paper/paper_spec.yaml
	•	Commande et lock : paper/LOCK.md
	•	Sortie : paper/out/<paper_id>/ (par défaut : paper/out/hc_taildep_v0_camera_ready/)
	•	Manifest : paper/out/<paper_id>/manifest.json

## Sorties attendues

Par défaut (voir paper/paper_spec.yaml), le paper_id est :

	•	hc_taildep_v0_camera_ready

La sortie principale est :
	•	paper/out/hc_taildep_v0_camera_ready/

Ce répertoire contient :
	•	tables/ : tables camera-ready (CSV stabilisés).
	•	figures/ : figures camera-ready (PNG/PDF selon la spec).
	•	manifest.json : manifest d’audit (intrants/sortants + hashes).

Un récapitulatif synthétique peut être disponible sous :
	•	paper/out/hc_taildep_v0_camera_ready/paper_summary.md

## Intrants et données

Les chemins des runs sources (J6/J7/J8/…) sont définis dans paper/paper_spec.yaml. Chaque répertoire de run est attendu avec un socle minimal d’auditabilité, typiquement :
	•	config.resolved.yaml
	•	provenance.json

et, selon les blocs, des tables/CSV (scores, paramètres, DM summaries, var_es_predictions.csv, etc.).

Les datasets et sorties intermédiaires sous data/processed/ peuvent être volumineux ; ce dépôt versionne le code, les spécifications et les scripts de build, mais ne versionne pas par défaut les données.

## Convention d’usage

Ce dépôt est orienté “mesure et preuve” : la lecture des résultats doit rester descriptive et prudente. Les diagnostics produits (tables/figures) visent à caractériser comparabilité, stabilité et contraintes de puissance, et non à soutenir une interprétation causale ou une promesse de performance.

## Statut

Dépôt de recherche et de production d’artefacts auditables. Les éléments sensibles ou volumineux (données, sorties lourdes) sont provisionnés séparément.


