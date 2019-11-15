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

TODO: validation/training thing: should we make it so augmentations are only applied while training?

Example runs:

```python main.py```