# Dataset

The dataset is available at:

http://www.cvlibs.net/datasets/kitti/raw_data.php

We used synced and rectified images.

The main videos used during developement are:

- 2011_09_26_drive_0005 (0.6 GB): http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip
- 2011_09_26_drive_0018 (1.1 GB): http://kitti.is.tue.mpg.de/kitti/raw_data/2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip


Color images are used (coming from camera 2 or 3).
In a generic dataset folder they can be found in `2011_**_**_drive_0***_sync/image_0[2-3]/data`

Timestamps can be found in `2011_**_**_drive_0***_sync/image_0[2-3]/timestamps.txt`

OXTS data is situated in `2011_**_**_drive_0***_sync/oxts`
In the OXTS folder are stored the GPS data timestamps (timestamps.txt), the data format (dataformat.txt) and under `data/` the informations collected at each datapoint sample (each in a different file)
