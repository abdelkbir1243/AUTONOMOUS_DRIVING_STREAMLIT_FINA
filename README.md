# Autonomous Driving Vision Lab — Streamlit

Application de démonstration du détecteur final du PFE :

`Driving-Aware I-JEPA → ViT-S/16 → Simple Feature Pyramid P2-P6 → Faster R-CNN`

Classes : **Pedestrian, Cyclist, Car, Van**. Entrée : **1024 × 320**.

## Pourquoi l'ancien dépôt ne fonctionnait pas

Le checkpoint Colab complet contient généralement l'optimiseur et peut être
beaucoup trop volumineux pour GitHub. De plus, un pointeur Git LFS non téléchargé
n'est pas un modèle utilisable. Ce projet fournit `prepare_checkpoint.py` :
il conserve uniquement `model_state_dict`, stocke les tenseurs flottants en
FP16 puis découpe le fichier en morceaux de 20 MiB. Streamlit les reconstitue
automatiquement et PyTorch recopie les poids dans le modèle FP32.

## Préparation dans Google Colab

1. Téléchargez ce ZIP et importez `PREPARER_ZIP_FINAL_COLAB.ipynb` dans Colab.
2. Dans le notebook, importez aussi le ZIP quand Colab le demande.
3. Montez Google Drive, vérifiez le chemin du checkpoint, puis exécutez tout.
4. Colab télécharge `AUTONOMOUS_DRIVING_STREAMLIT_FINAL.zip`.

Le chemin prévu dans le notebook est :

```text
/content/drive/MyDrive/PFE_JEPA/NB6_FINAL_COLAB/detection_E2_architecture/checkpoints/best_detector.pt
```

## Envoi sur GitHub

**N'envoyez pas le fichier ZIP lui-même dans le dépôt.** Décompressez
`AUTONOMOUS_DRIVING_STREAMLIT_FINAL.zip`, puis envoyez son contenu. À la racine
du dépôt GitHub, on doit voir directement :

```text
app.py
model.py
requirements.txt
models/best_detector.pt.part001
models/best_detector.pt.part002
...
.streamlit/config.toml
```

Les morceaux restent sous la limite de l'interface web GitHub. Aucun Git LFS
n'est nécessaire.

## Déploiement sur Streamlit Community Cloud

1. Ouvrez <https://share.streamlit.io/> et cliquez sur **Create app**.
2. Sélectionnez le dépôt GitHub et la branche `main`.
3. Main file path : `app.py`.
4. Dans **Advanced settings**, choisissez **Python 3.12**.
5. Cliquez sur **Deploy**.

Le premier démarrage installe PyTorch CPU. La première inférence charge le
modèle et peut prendre plusieurs dizaines de secondes sur l'offre gratuite.

## Test local facultatif

```bash
python -m pip install -r requirements.txt
python validate_deployment.py
streamlit run app.py
```

## Architecture prise en charge

Le chargeur reproduit les noms de modules du notebook E2 final et charge le
checkpoint avec `strict=True`. Il reconnaît également l'ancienne variante NB6
présente dans `notebook6(3).ipynb`.
