README

Il dataset è disponibile all'indirizzo:

http://www.cvlibs.net/datasets/kitti/raw_data.php

Si prendono le immagini sincronizzate e rettificate.

I principali video utilizzati nella sperimentazione sono:

- 2011_09_26_drive_0005 (0.6 GB): http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
- 2011_09_26_drive_0018 (1.1 GB): http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip


Le immagini utilizzate sono quelle a colori, quindi provenienti dalla camera 2 o 3.
In una generica cartella del dataset sono situate in 2011_**_**_drive_0***_sync/image_0[2-3]/data

I timestamps sono situati in 2011_**_**_drive_0***_sync/image_0[2-3]/timestamps.txt

I dati oxts sono situati in 2011_**_**_drive_0***_sync/oxts

Nella cartella oxts sono presenti i timestamps dei dati GPS (timestamps.txt), il formato dei dati (dataformat.txt) e in 'data/' le informazioni ad ogni istante di rilevazione in un differente file.

Insieme a questo documento si sono allegati per comodità anche il paper originale citato riguardo al dataset raw e lo schema di disposizione dei sensori nella macchina.