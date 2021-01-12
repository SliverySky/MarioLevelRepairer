# A Novel CNet-assisted Evolutionary Level Repairer  
'''T. Shu, Z. Wang, J. Liu and X. Yao, "A Novel CNet-assisted Evolutionary Level Repairer and Its Applications to Super Mario Bros," 2020 IEEE Congress on Evolutionary Computation (CEC), Glasgow, United Kingdom, 2020, pp. 1-10, doi: 10.1109/CEC48606.2020.9185538.
Applying latent variable evolution to game level design has become more and more popular as little human expert knowledge is required. However, defective levels with illegal
patterns may be generated due to the violation of constraints forlevel design. A traditional way of repairing the defective levelsis programming specific rule-based repairers to patch the flaw.'''

However, programming these constraints is sometimes complex and not straightforward. An autonomous level repairer which is capable of learning the constraints is needed. In this project, we use a novel approach, CNet, to learn the probability distribution of tiles giving its surrounding tiles on a set of real levels, and then detect the illegal tiles in generated new levels. Then, an evolutionary repairer is designed to search for optimal replacement schemes equipped with a novel search space being constructed with the help of CNet and a novel heuristic function. The proposed approaches are proved to be effective in our case study of repairing GAN-generated and artificially destroyed levels of Super Mario Bros. game. Our CNet-assisted evolutionary repairer can also be easily applied to other games of which the levels can be represented by a matrix of objects or tiles.  

For more information, please see the following publication. This publication should also be cited if code from this project is used in any way:

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

You can also find the pdf for free from https://arxiv.org/abs/2005.06148

### Requirements

Requirments to run this project and their version we used are as follow:

​	python 	(3.7.6)

​	numpy 	(1.18.1)

​	pytorch	(1.5.0)

​	pygame    (2.0.1)

### How to use

1. **Generate Data for CNet**: Run "*CNet/data/generate.py*".
2. **Train/Test CNet**: Run "*CNet/model.py*" to train CNet and run "*CNet/test.py*" to test the model. Please make sure the data is generated already.
3. **Generate levels to repair**: You can run "RandomDestroyed/generate.py" to generate random destroyed levels for the further test. As we mension in the paper, GAN will generate broken pipes sometimes. We prepared a trained GAN model in "*LevelGeneratir/GAN/generator.pth*". You can run "*LevelGenerator/generate_level.py*"  to generate some levels by GAN to see how many defective level will it generate.  
4. **Run GA to repair level**: Run "*GA/run.py*" to repair a defective level. The best invidivual at each epoch will be saved as an image in "*GA/result*" folder. In addition, you can run "*GA/clear.py*" to clean up the old results.
5. **Result Visulization:** Run "draw_graph.py" to draw the graph from repair results. Run "evaluate.py" to see how many true(wrong) tiles was changed to true(wrong) tiles after repair. Waht's more, you can run "render.py" to see the visuliized repair progress.
