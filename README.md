# HAF in PyTorch

Official Code Release for **[Learning Hierarchy Aware Features for Reducing Mistake Severity]** <br/>
Ashima Garg, Depanshu Sani, Saket Anand. <br />

_European Conference on Computer Vision (ECCV 2022)_


## Proposed Approach 
<div align="center">
  <img src="imgs/teaser_hist.png"/>
</div>


## Installation
Clone this repository
```
$ git clone https://github.com/07Agarg/HAF.git
$ cd HAF
```

## Using the Code
The experiments in the paper are contained in the folder ```experiments/``` dataset wise. 
For CIFAR-100
```
bash experiments/train/cifar-100/cross-entropy.sh
```
For iNaturalist-19
```
bash experiments/train/inat/cross-entropy.sh
```
For tiered-imagenet
```
bash experiments/train/tieredimagenet/cross-entropy.sh
```


## Acknowledgements
This codebase is borrowed from [making-better-mistakes](https://github.com/fiveai/making-better-mistakes)


## Contact 
If you have any suggestion or question, you can leave a message here or contact us directly at ashimag@iiitd.ac.in