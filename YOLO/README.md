README

Le dipendenze del progetto sono i seguenti pacchetti (moduli) Python:
• Pillow
• Numpy
• GeoPy
• Keras
• TensorFlow

Lo script principale è YOLOVideo.py

Un esempio di comando può essere:

$ python3 YOLOVideo.py -i <InputFolder> -o <OutputFolder>

La cartella di input deve essere una cartella che contiene in maniera sequenziale una successione di immagini corrispondenti ai frame video del dataset.

Si possono utilizzare anche dei parametri per personalizzare l'esecuzione:

$ python3 YOLOVideo.py -i <InputFolder> -o <OutputFolder> [ -m <PathToWeights> ] [ -a <AnchoresFilePath> ] [ -c <ClassesFilePath> ] [ -ox <PathToOxtsFolder> ] [ -ts <PathToTimestampsFile> ] [ -s <ScoreThreshold> ] [ -iou <IOUThreshold> ]

Per il significato dei vari argomenti vedere la relazione.

I dati raccolti (in una struttura dati Python) vengono salvati in un file pickle per poter essere utilizzati anche successivamente.

Le traiettorie possono essere mostrate graficamente utilizzando lo script plot.py che non richiede alcun parametro e utilizza i file pickle generati dallo script YOLOVideo.py
Per far avanzare lo script, premere qualunque tasto. Un avanzamento corrisponde ad un frame video.
Nel grafico ogni colore corrisponde ad un oggetto identificato diverso.