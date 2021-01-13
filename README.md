# A Novel CNet-assisted Evolutionary Level Repairer  
```T. Shu, Z. Wang, J. Liu and X. Yao, "A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, United Kingdom, 2020, pp. 1-10, doi: 10.1109/CEC48606.2020.9185538.```

## Brief introduction
Applying latent variable evolution to game level design has become more and more popular as little human expert knowledge is required. However, defective levels with illegal patterns may be generated due to the violation of constraints for level design. A traditional way of repairing the defective levels is implementing specific rule-based repairers to patch the flaw. However, implementing these constraints is sometimes complex and not straightforward. An autonomous level repairer which is capable of learning the constraints is needed. In this project, we consider tile-based game level design. We propose a novel approach, CNet, to learn the probability distribution of tiles giving its surrounding tiles on a set of real levels, and then detect the illegal tiles in the generated new levels. Then, an evolutionary repairer is designed to search for optimal replacement schemes equipped with a novel search space being constructed with the help of CNet and a novel heuristic function. The proposed approaches are proved to be effective in our case study of repairing GAN-generated and artificially destroyed levels of Super Mario Bros. game. Our CNet-assisted evolutionary repairer can also be easily applied to other games of which the levels can be represented by a matrix of elements or tiles.  

For more information, please see the following paper. This paper should also be cited if code from this project is used in any way:

```
@INPROCEEDINGS{shu2020novel,
  title={A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros},
  author={Shu, Tianye and Wang, Ziqi and Liu, Jialin and Yao, Xin},
  booktitle={2020 IEEE Congress on Evolutionary Computation (CEC)}, 
  doi={10.1109/CEC48606.2020.9185538},
  pages={1-10},
  year={2020}
}
```

You can also find the pdf of this paper for free on ArXiv: https://arxiv.org/abs/2005.06148

## Requirements

Requirments to run this project and the tested version are as follow:

* python (3.7.6)

* numpy (1.18.1)

* pytorch (1.5.0)

* pygame (2.0.1)

## How to use
### Quick start without re-training
The generated data and trained model is given in this project. You can run the repairer and visualise the results directly by the following steps.
* **Run GA to repair level**: Run ***GA/run.py*** to repair a defective level. The best invidivual at each epoch will be saved as an image in "*GA/result*" folder. In addition, you can run "*GA/clear.py*" to clean up the old results.

```python GA/run.py```

```python GA/clear.py```

* **Result Visulization:** Run ***draw_graph.py*** to draw the graph from repair results. Run ***evaluate.py*** to see how many true(wrong) tiles was changed to true(wrong) tiles after repair. What's more, you can run "render.py" to see the visuliized repair progress.

```python draw_graph.py```

### Start from the very beginning (data generation, model training)
* **Generate Data for CNet**: 

```python CNet/data/generate.py```

The generated data will be located in the folder ***./CNet/data/***.

* **Train/Test CNet**: Run ***CNet/model.py*** to train CNet and run ***CNet/test.py*** to test the model. Please make sure the data is generated already.

```python CNet/model.py```

```python CNet/test.py```

* **Generate levels to repair**: You can run ***RandomDestroyed/generate.py*** to generate random destroyed levels for the further test. As we mension in the paper, GAN will generate broken pipes sometimes. We prepared a trained GAN model in ***LevelGeneratir/GAN/generator.pth***. You can run ***LevelGenerator/generate_level.py*** to generate some levels by GAN to see how many defective level will it generate.  

```python RandomDestroyed/generate.py```

```python LevelGeneratir/GAN/generator.pth```

* **Run GA to repair level**: Run ***GA/run.py*** to repair a defective level. The best invidivual at each epoch will be saved as an image in "*GA/result*" folder. In addition, you can run "*GA/clear.py*" to clean up the old results.

```python GA/run.py```

```python GA/clear.py```

* **Result Visulization:** Run ***draw_graph.py*** to draw the graph from repair results. Run ***evaluate.py*** to see how many true(wrong) tiles was changed to true(wrong) tiles after repair. What's more, you can run "render.py" to see the visuliized repair progress.

```python draw_graph.py```

## Project structure
* **Assets/Tiles**:
* **CNET** folder:
* **GA** folder:
* **LevelGenerator** folder:
* **LevelText** folder:
* **utiles** folder:
* **root.py**:
* **test.py**:

.
├── Assets
│   ├── Tiles
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── 10.jpg
│   │   ├── 2.jpg
│   │   ├── 3.jpg
│   │   ├── 4.jpg
│   │   ├── 5.jpg
│   │   ├── 6.jpg
│   │   ├── 7.jpg
│   │   ├── 8.jpg
│   │   └── 9.jpg
│   └── __init__.py
├── CNet
│   ├── CNet_old.pkl
│   ├── __init__.py
│   ├── __pycache__
│   │   ├── __init__.cpython-37.pyc
│   │   └── model.cpython-37.pyc
│   ├── data
│   │   ├── __init__.py
│   │   ├── all_elm_rule.json
│   │   ├── generate.py
│   │   ├── illegal_rule.json
│   │   ├── illegal_rule_F1.json
│   │   ├── illegal_rule_F2.json
│   │   ├── illegal_rule_F3.json
│   │   ├── legal_rule.json
│   │   ├── legal_rule_F1.json
│   │   ├── legal_rule_F2.json
│   │   └── legal_rule_F3.json
│   ├── dict.pkl
│   ├── model.py
│   ├── rule_fake.json
│   └── test.py
├── GA
│   ├── __init__.py
│   ├── __pycache__
│   │   └── repair.cpython-37.pyc
│   ├── best.png
│   ├── clear.py
│   ├── draw_graph.py
│   ├── evaluate.py
│   ├── mean.png
│   ├── point.png
│   ├── render.py
│   ├── repair.py
│   ├── result
│   │   ├── figure
│   │   │   ├── iteration0.jpg
│   │   │   ├── iteration1.jpg
│   │   │   ├── iteration10.jpg
│   │   │   ├── iteration11.jpg
│   │   │   ├── iteration12.jpg
│   │   │   ├── iteration13.jpg
│   │   │   ├── iteration14.jpg
│   │   │   ├── iteration15.jpg
│   │   │   ├── iteration16.jpg
│   │   │   ├── iteration17.jpg
│   │   │   ├── iteration18.jpg
│   │   │   ├── iteration19.jpg
│   │   │   ├── iteration2.jpg
│   │   │   ├── iteration20.jpg
│   │   │   ├── iteration21.jpg
│   │   │   ├── iteration22.jpg
│   │   │   ├── iteration23.jpg
│   │   │   ├── iteration24.jpg
│   │   │   ├── iteration25.jpg
│   │   │   ├── iteration26.jpg
│   │   │   ├── iteration27.jpg
│   │   │   ├── iteration28.jpg
│   │   │   ├── iteration29.jpg
│   │   │   ├── iteration3.jpg
│   │   │   ├── iteration30.jpg
│   │   │   ├── iteration31.jpg
│   │   │   ├── iteration32.jpg
│   │   │   ├── iteration33.jpg
│   │   │   ├── iteration34.jpg
│   │   │   ├── iteration35.jpg
│   │   │   ├── iteration36.jpg
│   │   │   ├── iteration37.jpg
│   │   │   ├── iteration38.jpg
│   │   │   ├── iteration39.jpg
│   │   │   ├── iteration4.jpg
│   │   │   ├── iteration40.jpg
│   │   │   ├── iteration41.jpg
│   │   │   ├── iteration42.jpg
│   │   │   ├── iteration43.jpg
│   │   │   ├── iteration44.jpg
│   │   │   ├── iteration45.jpg
│   │   │   ├── iteration46.jpg
│   │   │   ├── iteration47.jpg
│   │   │   ├── iteration48.jpg
│   │   │   ├── iteration49.jpg
│   │   │   ├── iteration5.jpg
│   │   │   ├── iteration50.jpg
│   │   │   ├── iteration6.jpg
│   │   │   ├── iteration7.jpg
│   │   │   ├── iteration8.jpg
│   │   │   └── iteration9.jpg
│   │   ├── json
│   │   │   └── data.json
│   │   ├── result(Remark).jpg
│   │   ├── result.jpg
│   │   ├── result.txt
│   │   ├── start(Remark).jpg
│   │   ├── start.jpg
│   │   ├── start.txt
│   │   └── txt
│   │       ├── iteration0.txt
│   │       ├── iteration1.txt
│   │       ├── iteration10.txt
│   │       ├── iteration11.txt
│   │       ├── iteration12.txt
│   │       ├── iteration13.txt
│   │       ├── iteration14.txt
│   │       ├── iteration15.txt
│   │       ├── iteration16.txt
│   │       ├── iteration17.txt
│   │       ├── iteration18.txt
│   │       ├── iteration19.txt
│   │       ├── iteration2.txt
│   │       ├── iteration20.txt
│   │       ├── iteration21.txt
│   │       ├── iteration22.txt
│   │       ├── iteration23.txt
│   │       ├── iteration24.txt
│   │       ├── iteration25.txt
│   │       ├── iteration26.txt
│   │       ├── iteration27.txt
│   │       ├── iteration28.txt
│   │       ├── iteration29.txt
│   │       ├── iteration3.txt
│   │       ├── iteration30.txt
│   │       ├── iteration31.txt
│   │       ├── iteration32.txt
│   │       ├── iteration33.txt
│   │       ├── iteration34.txt
│   │       ├── iteration35.txt
│   │       ├── iteration36.txt
│   │       ├── iteration37.txt
│   │       ├── iteration38.txt
│   │       ├── iteration39.txt
│   │       ├── iteration4.txt
│   │       ├── iteration40.txt
│   │       ├── iteration41.txt
│   │       ├── iteration42.txt
│   │       ├── iteration43.txt
│   │       ├── iteration44.txt
│   │       ├── iteration45.txt
│   │       ├── iteration46.txt
│   │       ├── iteration47.txt
│   │       ├── iteration48.txt
│   │       ├── iteration49.txt
│   │       ├── iteration5.txt
│   │       ├── iteration50.txt
│   │       ├── iteration6.txt
│   │       ├── iteration7.txt
│   │       ├── iteration8.txt
│   │       └── iteration9.txt
│   └── run.py
├── LevelGenerator
│   ├── GAN
│   │   ├── Destroyed
│   │   │   ├── lv0.jpg
│   │   │   ├── lv0.txt
│   │   │   ├── lv1.jpg
│   │   │   ├── lv1.txt
│   │   │   ├── lv2.jpg
│   │   │   ├── lv2.txt
│   │   │   ├── lv3.jpg
│   │   │   ├── lv3.txt
│   │   │   ├── lv4.jpg
│   │   │   └── lv4.txt
│   │   ├── __pycache__
│   │   │   └── dcgan.cpython-37.pyc
│   │   ├── dcgan.py
│   │   ├── generate_level.py
│   │   └── generator.pth
│   ├── RandomDestroyed
│   │   ├── generate.py
│   │   ├── lv0.jpg
│   │   ├── lv0.txt
│   │   ├── lv1.jpg
│   │   ├── lv1.txt
│   │   ├── lv2.jpg
│   │   ├── lv2.txt
│   │   ├── lv3.jpg
│   │   ├── lv3.txt
│   │   ├── lv4.jpg
│   │   ├── lv4.txt
│   │   ├── lv5.jpg
│   │   ├── lv5.txt
│   │   ├── lv6.jpg
│   │   ├── lv6.txt
│   │   ├── lv7.jpg
│   │   ├── lv7.txt
│   │   ├── lv8.jpg
│   │   ├── lv8.txt
│   │   ├── lv9.jpg
│   │   └── lv9.txt
│   └── __init__.py
├── LevelText
│   ├── MarioBrother2
│   │   ├── SuperMarioBros2(J)-World1-1.txt
│   │   ├── SuperMarioBros2(J)-World1-2.txt
│   │   ├── SuperMarioBros2(J)-World1-3.txt
│   │   ├── SuperMarioBros2(J)-World2-1.txt
│   │   ├── SuperMarioBros2(J)-World2-2.txt
│   │   ├── SuperMarioBros2(J)-World2-3.txt
│   │   ├── SuperMarioBros2(J)-World3-3.txt
│   │   ├── SuperMarioBros2(J)-World4-1.txt
│   │   ├── SuperMarioBros2(J)-World4-3.txt
│   │   ├── SuperMarioBros2(J)-World5-2.txt
│   │   ├── SuperMarioBros2(J)-World6-1.txt
│   │   ├── SuperMarioBros2(J)-World6-3.txt
│   │   ├── SuperMarioBros2(J)-WorldA-1.txt
│   │   ├── SuperMarioBros2(J)-WorldA-3.txt
│   │   ├── SuperMarioBros2(J)-WorldB-1.txt
│   │   ├── SuperMarioBros2(J)-WorldB-3.txt
│   │   └── SuperMarioBros2(J)-WorldC-2.txt
│   ├── __init__.py
│   └── pipes.txt
├── README.md
├── __init__.py
├── __pycache__
│   ├── root.cpython-36.pyc
│   └── root.cpython-37.pyc
├── root.py
├── test.py
└── utils
    ├── __init__.py
    ├── __pycache__
    │   ├── __init__.cpython-36.pyc
    │   ├── level_process.cpython-36.pyc
    │   ├── level_process.cpython-37.pyc
    │   └── visualization.cpython-37.pyc
    ├── level_process.py
    └── visualization.py

21 directories, 219 files
