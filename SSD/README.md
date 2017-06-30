README

Il codice si trova all'interno della cartella 'src' .
La cartella 'trained_models' contiene invece i pesi del modello SSD.

Le dipendenze del progetto sono i seguenti pacchetti (moduli) Python:
• Pillow
• Numpy
• GeoPy
• Keras
• TensorFlow

Lo script principale è SSDVideo.py

Un esempio di comando può essere:

$ python3 SSDVideo.py -i <InputFolder> -o <OutputFolder>

La cartella di input deve essere una cartella che contiene in maniera sequenziale una successione di immagini corrispondenti ai frame video del dataset.

Si possono utilizzare anche dei parametri per personalizzare l'esecuzione:

$ python3 SSDVideo.py -i <InputFolder> -o <OutputFolder> [ -m <PathToWeights> ] [ -ox <PathToOxtsFolder> ] [ -ts <PathToTimestampsFile> ]

Per il significato dei vari argomenti vedere la relazione.

I dati raccolti (in una struttura dati Python) vengono salvati in un file pickle per poter essere utilizzati anche successivamente.

Le traiettorie possono essere mostrate graficamente utilizzando lo script plot.py che non richiede alcun parametro e utilizza i file pickle generati dallo script SSDVideo.py
Per far avanzare lo script, premere qualunque tasto. Un avanzamento corrisponde ad un frame video.
Nel grafico ogni colore corrisponde ad un oggetto identificato diverso.