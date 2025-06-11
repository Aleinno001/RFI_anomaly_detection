# Setup environment

Il processo di setup è fatto in modo da essere compatibile al massimo con il software sviluppato da UniSa.

1. Creare ambiente conda

```bash
conda create -n RFI_anomaly_detection_SAM python=3.10
conda activate RFI_anomaly_detection_SAM
```

2. Installare le dipendenze, tra cui CUDA, CuDNN e librerie per Deep Learning:

Installare SAM2 e Grounding DINO:

```bash
pip install sam2
pip install rf-groundingdino
```
Il pacchetto Grounding Dino non è ufficiale ma è mantenuto da Roboflow e semplifica l'installazione rispetto alla versione sorgente origianle.
Nota: la versione RF-GroundingDINO ha una dipendenza da supervision senza specificarne la versione ma il pacchetto Pip è stato creato quando la versione era la 0.21 e ne dipende.
Per far funzionare la visulizzazione si deve quindi fare un downgrade di supervision con:

```bash
pip install -r requirements.txt
```

o in alternativa con:

```bash
pip install supervision==0.21.0
```

3. Scaricare modelli:

Grounding DINO con wget:

```bash
wget https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth
```

SAM2 con wget:

```bash
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt
```

Spostare i modelli nelle rispettive cartelle:

```bash
mv groundingdino_swint_ogc.pth models/groundingdino/
mv sam2.1_hiera_tiny.pt models/sam2.1/
``` 