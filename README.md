# hc_taildep : Tail Dependence & Regime-Aware Copula Evaluation (BTC/ETH)

## Objet

Ce dépôt contient une chaîne de recherche et de production d’artefacts visant une évaluation reproductible de la dépendance de queue (tail dependence) et de ses régimes sur un univers crypto (BTC/ETH en priorité). L’objectif n’est pas de “prédire le marché”, mais de produire des mesures opposables, auditables et reconstructibles, utilisables comme composant de gouvernance du risque (contrôles de stabilité, diagnostics de puissance, fallbacks, et preuves associées).

La philosophie générale est contractuelle : à partir de données et de configurations versionnées, la pipeline doit reconstruire bit-à-bit les tables, figures et manifests décrits dans le papier. Le dépôt est structuré pour permettre une revue externe (reproductibilité, traçabilité, absence d’ajustements manuels ex post).

## Périmètre scientifique (résumé)

Le dépôt couvre la mesure de la dépendance de queue sous différentes hypothèses de copules (t-copula et variantes), l’analyse de régimes (calm / stress, ou variantes), la comparaison de politiques (static / threshold / logistic gating / markov switching), ainsi que des diagnostics “impact risk” via VaR/ES et tests de calibration (exceedance, Kupiec POF).

Les sorties prioritaires sont des artefacts (CSV/PNG/JSON) et non un texte narratif : chaque section du papier renvoie à des preuves matérialisées par des tables et figures, et chaque run génère un manifest contenant les empreintes SHA-256 des intrants/sortants.

## Organisation du dépôt

Le code applicatif et la logique de calcul résident sous `src/hc_taildep/`. La génération des artefacts “papier” est sous `paper/`. Les datasets et sorties intermédiaires sont sous `data/processed/` et doivent être considérés comme des intrants de recherche (potentiellement volumineux) ; par défaut, ils ne sont pas versionnés dans Git.

La sortie “camera-ready” (tables, figures, manifest) est écrite sous `paper/out/<paper_id>/` et constitue l’interface principale pour toute revue.

## Environnement d’exécution

Le dépôt est conçu pour être exécuté dans un environnement Python isolé (venv). Les figures sont rendues en backend non-interactif (Agg) afin d’assurer une exécution stable en CI ou sur serveur.

Installation typique :

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
