TODO: train pytorch model on anton new data?
TODO: add transformations, image normalization and augmentations?
TODO: use validation set
TODO: use more cmd args? to set lr etc.?
TODO: save model
TODO: test dataset from without_noise?
TODO: check jyri mäenpää augmentation and do the same maybe?
TODO: test in CARLA simulation which works better with noise or without?
TODO: katso poikkeamat oikeasta steeristä, onko systemaattisia erroreja esim vasemmalle? plottaa virheitä. standard error jne. variance, worst case scenariot. katso segmentointi kuvaa ja katso korreloiko jokin tietty muoto erroriin (esim puita paljon). testaa test setissä aikajärjestyksessä että alkaako steer väärään, vai onko possible, koska test setissä ei ole kuvia jossa se veer off?
TODO: kysy antonilta tarkemmin carla mallista, osaa esim. liikennevalot?
TODO: create config with folders where data is stored, models are stored, and so on? apparently /l is the local disk
TODO: carla from source? needed?
TODO: test dataset? without noise? plot errors jne?

TODO: validation/training thing: should we make it so augmentations are only applied while training?
TODO 19.11.2019: create_dataset_new, miten indeksit jne?

dataset pelleilyt:

- create_dataset_new.py: uuden kaltaiset, eri indeksit
- create_dataset.py: vanhat, with_noisessa paljon dataa joten kantsii
- many_datasets.sh - ottaa ne kaikki
- sit viela merge_datasets.py: mergaa ne create_datasetin jalkeiset yhteen folderiin kun tarvii teha ehka?
- epailyt: jossakin vaiheessa oli eri maara imageja ja ster angleja mebbe? en oo varma oliko oma virhe.

TODO kun saa datasetit kuntoon: voi trainata kaikilla, ehka testaa without_noisella?

Create video with ffmpeg: ```ffmpeg -start_number 0 -i %d.jpg -vcodec mpeg4 -framerate 60 test.avi```

Example runs:

```python main.py```

Carla example:

start carla simulation (for example in CarlaUE4 -> binaries -> Carla...)
run example.py while carla simulation is running