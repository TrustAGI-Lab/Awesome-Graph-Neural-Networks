## Awesome resources on Graph Neural Networks.
This is a collection of resources related with graph neural networks.


## Contents

- [Survey papers](#surveypapers)
- [Papers](#papers)
  - [Recuurent Graph Neural Networks](#rgnn)
  - [Convolutional Graph Neural Networks](#cgnn)
  - [Graph Autoencoders](#gae)
  	  - [Network Embedding](#ne)
  	  - [Graph Generation](#gg)
  - [Spatial-Temporal Graph Neural Networks](#stgnn)
  - [Application](#application)
     - [Computer Vision](#cv)
     - [Natural Language Processing](#nlp)
     - [Internet](#web)
     - [Recommender Systems](#rec)
     - [Healthcare](#health)
     - [Chemistry](#chemistry)
     - [Physics](#physics)
     - [Others](#others)
- [Library](#library)
<a name="surveypapers" />

## Survey papers

1. **A Comprehensive Survey on Graph Neural Networks.** *Zonghan Wu, Shirui Pan, Fengwen Chen, Guodong Long, Chengqi Zhang, Philip S. Yu.* 2019 [paper](https://arxiv.org/pdf/1901.00596.pdf)

1. **Adversarial Attack and Defense on Graph Data: A Survey.** *Lichao Sun, Yingtong Dou, Carl Yang, Ji Wang, Philip S. Yu, Bo Li.* 2018 [paper](https://arxiv.org/pdf/1812.10528.pdf)

1. **Geometric deep learning: going beyond euclidean data.** *Michael M. Bronstein, Joan Bruna, Yann LeCun, Arthur Szlam, Pierre Vandergheynst.*  2016. [paper](https://arxiv.org/pdf/1611.08097.pdf)

1. **Relational inductive biases, deep learning, and graph networks.**
*Peter W. Battaglia, Jessica B. Hamrick, Victor Bapst, Alvaro Sanchez-Gonzalez, Vinicius Zambaldi, Mateusz Malinowski, Andrea Tacchetti, David Raposo, Adam Santoro, Ryan Faulkner, Caglar Gulcehre, Francis Song, Andrew Ballard, Justin Gilmer, George Dahl, Ashish Vaswani, Kelsey Allen, Charles Nash, Victoria Langston, Chris Dyer, Nicolas Heess, Daan Wierstra, Pushmeet Kohli, Matt Botvinick, Oriol Vinyals, Yujia Li, Razvan Pascanu.* 2018. [paper](https://arxiv.org/pdf/1806.01261.pdf)

1. **Attention models in graphs.** *John Boaz Lee, Ryan A. Rossi, Sungchul Kim, Nesreen K. Ahmed, Eunyee Koh.* 2018. [paper](https://arxiv.org/pdf/1807.07984.pdf)

1. **Deep learning on graphs: A survey.** Ziwei Zhang, Peng Cui and Wenwu Zhu. 2018. [paper](https://arxiv.org/pdf/1812.04202.pdf)

1. **Graph Neural Networks: A Review of Methods and Applications** *Jie Zhou, Ganqu Cui, Zhengyan Zhang, Cheng Yang, Zhiyuan Liu, Maosong Sun.* 2018 [paper](https://arxiv.org/pdf/1812.08434.pdf)



<a name="papers" />

## Papers

<a name="rgnn" />

## Recurrent Graph Neural Networks
1. **Supervised neural networks for the classification of structures** *A. Sperduti and A. Starita.* IEEE Transactions on Neural Networks 1997. [paper](https://www.ncbi.nlm.nih.gov/pubmed/18255672)

1. **A new model for learning in graph domains.** *Marco Gori, Gabriele Monfardini, Franco Scarselli.* IJCNN 2005. [paper](https://ieeexplore.ieee.org/abstract/document/1555942)

1. **The graph neural network model.** *Franco Scarselli,Marco Gori,Ah Chung Tsoi,Markus Hagenbuchner,
Gabriele Monfardini.* 2009. [paper](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1015.7227&rep=rep1&type=pdf)

1. **Graph echo state networks.** *Claudio Gallicchio, Alessio Micheli* IJCNN 2010. [paper](https://ieeexplore.ieee.org/abstract/document/5596796)

1. **Gated graph sequence neural networks.** *Yujia Li, Richard Zemel, Marc Brockschmidt, Daniel Tarlow.* ICLR 2015. [paper](https://arxiv.org/pdf/1511.05493.pdf)

1. **Learning steady-states of iterative algorithms over graphs.** *Hanjun Dai, Zornitsa Kozareva, Bo Dai, Alexander J. Smola, Le Song* ICML 2018. [paper](http://proceedings.mlr.press/v80/dai18a/dai18a.pdf)

<a name="cgnn" />

## Convolutional Graph Neural Networks

### Spectral

1. **Spectral networks and locally connected networks on graphs.** *Joan Bruna, Wojciech Zaremba, Arthur Szlam, Yann LeCun.* ICLR 2014. [paper](https://arxiv.org/pdf/1312.6203.pdf)

1. **Deep convolutional networks on graph-structured data.** *Mikael Henaff, Joan Bruna, Yann LeCun.* 2015. [paper](https://arxiv.org/abs/1506.05163) 

1. **Accelerated filtering on graphs using lanczos method.** *Ana Susnjara, Nathanael Perraudin, Daniel Kressner, Pierre Vandergheynst.* 2015. [paper](https://arxiv.org/pdf/1509.04537.pdf)

1. **Convolutional neural networks on graphs with fast localized spectral filtering.** *Michaël Defferrard, Xavier Bresson, Pierre Vandergheynst.* NIPS 2016. [paper](https://arxiv.org/pdf/1606.09375.pdf)


1. **Semi-supervised classification with graph convolutional networks.** *Thomas N. Kipf, Max Welling.* ICLR 2017. [paper](https://arxiv.org/pdf/1609.02907.pdf)

1. **Cayleynets: graph convolutional neural networks with complex rational spectral filters.** *Ron Levie, Federico Monti, Xavier Bresson, Michael M. Bronstein.* 2017. [paper](https://arxiv.org/pdf/1705.07664.pdf)

1. **Simplifying Graph Convolutional Networks.** *Felix Wu, Tianyi Zhang, Amauri Holanda de Souza Jr., Christopher Fifty, Tao Yu, Kilian Q. Weinberger.* ICML 2019. [paper](https://arxiv.org/pdf/1902.07153.pdf) [code](https://github.com/Tiiiger/SGC)

1. **Graph Wavelet Neural Network.** *Bingbing Xu, Huawei Shen, Qi Cao, Yunqi Qiu, Xueqi Cheng.* ICLR 2019. [paper](https://openreview.net/pdf?id=H1ewdiR5tQ)


1. **DIFFUSION SCATTERING TRANSFORMS ON GRAPHS.** *Fernando Gama, Alejandro Ribeiro, Joan Bruna.* ICLR 2019. [paper](https://arxiv.org/pdf/1806.08829.pdf)

### Spatial

1. **Neural network for graphs: A contextual constructive approach.** *A. Micheli.*  IEEE Transactions on Neural Networks 2009. [paper](https://ieeexplore.ieee.org/abstract/document/4773279)

1. **Convolutional networks on graphs for learning molecular fingerprints.** *David Duvenaud, Dougal Maclaurin, Jorge Aguilera-Iparraguirre Rafael Go ́mez-Bombarelli, Timothy Hirzel, Ala ́n Aspuru-Guzik, Ryan P. Adams.*, NIPS 2015. [paper](http://papers.nips.cc/paper/5954-convolutional-networks-on-graphs-for-learning-molecular-fingerprints.pdf)


1. **Diffusion-convolutional neural networks** *James Atwood, Don Towsley.* NIPS 2016. [paper](https://arxiv.org/pdf/1511.02136.pdf)

1. **Neural message passing for quantum chemistry.** *Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl.* ICML 2017. [paper](https://arxiv.org/pdf/1704.01212.pdf)

1. **Dynamic Edge-Conditioned Filters in Convolutional Neural Networks on Graphs** *Martin Simonovsky, Nikos Komodakis* CVPR 2017. [paper](https://arxiv.org/pdf/1704.02901.pdf) 

1. **Geometric deep learning on graphs and manifolds using mixture model cnns.** *Federico Monti, Davide Boscaini, Jonathan Masci, Emanuele Rodolà, Jan Svoboda, Michael M. Bronstein.* CVPR 2017. [paper](https://arxiv.org/pdf/1611.08402.pdf)


1. **Robust spatial filtering with graph convolutional neural networks.** 2017. *Felipe Petroski Such, Shagan Sah, Miguel Dominguez, Suhas Pillai, Chao Zhang, Andrew Michael, Nathan Cahill, Raymond Ptucha.* [paper](https://arxiv.org/abs/1703.00792)


1. **Structure-Aware Convolutional Neural Networks.** *Jianlong Chang, Jie Gu, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, Chunhong Pan.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7287-structure-aware-convolutional-neural-networks.pdf) [code](https://github.com/vector-1127/SACNNs)


1. **On filter size in graph convolutional network.** *D. V. Tran, A. Sperduti et al.* SSCI. IEEE, 2018. [paper](https://arxiv.org/pdf/1811.10435.pdf)

1. **Predict then Propagate: Graph Neural Networks meet Personalized PageRank.** *Johannes Klicpera, Aleksandar Bojchevski, Stephan Günnemann.* ICLR 2019. [paper](https://arxiv.org/pdf/1810.05997.pdf) [code](https://github.com/benedekrozemberczki/APPNP)


#### Architecture

1. **Representation learning on graphs with jumping knowledge networks.** *Keyulu Xu, Chengtao Li, Yonglong Tian, Tomohiro Sonobe, Ken-ichi Kawarabayashi, Stefanie Jegelka.* ICML 2018. [paper](https://arxiv.org/pdf/1806.03536.pdf)

1. **Dual graph convolutional networks for graph-based semi-supervised classification** *Chenyi Zhuang, Qiang Ma.* WWW 2018. [paper](https://dl.acm.org/citation.cfm?id=3186116)

1. **Graph U-nets** *Hongyang Gao, Shuiwang Ji.* ICML 2019. [paper](https://arxiv.org/pdf/1905.05178.pdf) [code](https://github.com/HongyangGao/gunet)

1. **MixHop: Higher-Order Graph Convolutional Architectures via Sparsified Neighborhood Mixing.** *Sami Abu-El-Haija, Bryan Perozzi, Amol Kapoor, Nazanin Alipourfard, Kristina Lerman, Hrayr Harutyunyan, Greg Ver Steeg, Aram Galstyan.* [paper](https://arxiv.org/pdf/1905.00067.pdf) [code](github.com/samihaija/mixhop)

#### Attention/Gating Mechanisms 

1. **Graph Attention Networks.**
*Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio.* ICLR 2018. [paper](https://arxiv.org/pdf/1710.10903.pdf) [code](https://github.com/PetarV-/GAT)

1. **Gaan: Gated attention networks for learning on large and spatiotemporal graphs.** *Jiani Zhang, Xingjian Shi, Junyuan Xie, Hao Ma, Irwin King, Dit-Yan Yeung.* 2018. [paper](https://arxiv.org/pdf/1803.07294.pdf)

1. **Geniepath: Graph neural networks with adaptive receptive paths.** Ziqi Liu, Chaochao Chen, Longfei Li, Jun Zhou, Xiaolong Li, Le Song, Yuan Qi. AAAI 2019. [paper](https://arxiv.org/pdf/1802.00910.pdf)

1. **Graph Representation Learning via Hard and Channel-Wise Attention Networks.** *Hongyang Gao, Shuiwang Ji.* 2019 KDD. [paper](https://www.kdd.org/kdd2019/accepted-papers/view/graph-representation-learning-via-hard-and-channel-wise-attention-networks) 

1. **Understanding Attention and Generalization in Graph Neural Networks.** *Boris Knyazev, Graham W. Taylor, Mohamed R. Amer.* NeurIPS 2019. [paper](https://arxiv.org/abs/1905.02850)

#### Convolution 

1. **Learning convolutional neural networks for graphs.** *Mathias Niepert, Mohamed Ahmed, Konstantin Kutzkov.* ICML 2016. [paper](https://arxiv.org/pdf/1605.05273.pdf)

1. **Large-Scale Learnable Graph Convolutional Networks.** *Hongyang Gao, Zhengyang Wang, Shuiwang Ji.* KDD 2018. [paper](https://arxiv.org/pdf/1808.03965.pdf)


#### Training Methods

1. **Inductive representation learning on large graphs.** *William L. Hamilton, Rex Ying, Jure Leskovec.* NIPS 2017. [paper](http://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs.pdf)

1. **Stochastic Training of Graph Convolutional Networks with Variance Reduction.**
*Jianfei Chen, Jun Zhu, Le Song.* ICML 2018. [paper](https://arxiv.org/pdf/1710.10568.pdf)

1. **Adaptive Sampling Towards Fast Graph Representation Learning.** *Wenbing Huang, Tong Zhang, Yu Rong, Junzhou Huang.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1809.05343.pdf) [code]()

1. **FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling.**
*Jie Chen, Tengfei Ma, Cao Xiao.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.10247.pdf)

1. **Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks.** KDD 2019. [paper](https://arxiv.org/pdf/1905.07953.pdf) [code](https://github.com/google-research/google-research/tree/master/cluster_gcn)




#### Pooling 

1. **Hierarchical graph representation learning with differentiable pooling.** *Rex Ying, Jiaxuan You, Christopher Morris, Xiang Ren, William L. Hamilton, Jure Leskovec.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.08804.pdf) [code](https://github.com/RexYing/diffpool)

1. **Self-Attention Graph Pooling.**  *Junhyun Lee, Inyeop Lee, Jaewoo Kang.* ICML 2019. [paper](https://arxiv.org/abs/1904.08082) [code](https://github.com/inyeoplee77/SAGPool)

#### Graph Classfication

1. **Contextual graph markov model: A deep and generative approach to graph processing.** *D. Bacciu, F. Errica,  A. Micheli.* ICML 2018. [paper](https://arxiv.org/abs/1805.10636)

1. **Adaptive graph convolutional neural networks.** *Ruoyu Li, Sheng Wang, Feiyun Zhu, Junzhou Huang.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.03226.pdf)


1. **Graph capsule convolutional neural networks** *Saurabh Verma, Zhi-Li Zhang.* 2018. [paper](https://arxiv.org/abs/1805.08090)


1. **Capsule Graph Neural Network** *Zhang Xinyi, Lihui Chen.* ICLR 2019. [paper](https://openreview.net/pdf?id=Byl8BnRcYm)


#### Bayesian

1. **Bayesian Semi-supervised Learning with Graph Gaussian Processes
.** *Yin Cheng Ng, Nicolò Colombo, Ricardo Silva*  NeurIPS 2018. [paper](https://papers.nips.cc/paper/7440-bayesian-semi-supervised-learning-with-graph-gaussian-processes.pdf)

	> It redefines the kernel function in Gaussian Process with graph structure information. 

1. **Bayesian Graph Convolutional Neural Networks for Semi-supervised Classification** *Yingxue Zhang, Soumyasundar Pal, Mark Coates, Deniz Üstebay.* AAAI 2019. [paper] (https://arxiv.org/pdf/1811.11103.pdf)


#### Analysis

1. **Deeper insights into graph convolutional networks for semi-supervised learning.** *Qimai Li, Zhichao Han, Xiao-Ming Wu.* AAAI 2018. [paper](https://arxiv.org/pdf/1801.07606.pdf)

1. **How powerful are graph neural networks?** *Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka.* ICLR 2019. [paper](https://arxiv.org/pdf/1810.00826.pdf)

1. **Can GCNs Go as Deep as CNNs?.** *Guohao Li, Matthias Müller, Ali Thabet, Bernard Ghanem.* 2019. ICCV 2019. [paper](https://arxiv.org/abs/1904.03751)

1. **Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks.** *Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, Martin Grohe* AAAI 2019. [paper](https://arxiv.org/pdf/1810.02244.pdf)


#### Miscellaneous Graphs

1. **Modeling relational data with graph convolutional networks** *Michael Schlichtkrull, Thomas N. Kipf, Peter Bloem, Rianne van den Berg, Ivan Titov, Max Welling.* ESW 2018. [paper](https://arxiv.org/pdf/1703.06103.pdf)

1. **Signed graph convolutional network**. *Tyler Derr, Yao Ma, Jiliang Tang.* 2018. [paper](https://arxiv.org/pdf/1808.06354.pdf)

1. **Multidimensional graph convolutional networks** *Yao Ma, Suhang Wang, Charu C. Aggarwal, Dawei Yin, Jiliang Tang.* 2018. [paper](https://arxiv.org/pdf/1808.06099.pdf)

1. **LanczosNet: Multi-Scale Deep Graph Convolutional Networks** *Renjie Liao, Zhizhen Zhao, Raquel Urtasun, Richard Zemel.* ICLR 2019. [paper](https://openreview.net/pdf?id=BkedznAqKQ)

1. **Hypergraph Neural Networks.** *Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao* AAAI 2019. [paper](https://arxiv.org/pdf/1809.09401.pdf)

1. **HyperGCN: A New Method of Training Graph Convolutional Networks on Hypergraphs.** *
Naganand Yadati, Madhav Nimishakavi, Prateek Yadav, Vikram Nitin, Anand Louis, Partha Talukdar.* NeurIPS 2019. [paper](https://arxiv.org/abs/1809.02589)



<a name="gae" />


## Graph Auto-encoder

<a name="ne" />

### Network Embedding
1. **Structural deep network embedding** *Daixin Wang, Peng Cui, Wenwu Zhu.* [paper](https://www.kdd.org/kdd2016/papers/files/rfp0191-wangAemb.pdf)

1. **Deep neural networks for learning graph representations.** *Shaosheng Cao, Wei Lu, Qiongkai Xu.* AAAI 2016. [paper](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12423/11715)

1. **Variational graph auto-encoders.** *Thomas N. Kipf, Max Welling.* 2016. [paper](https://arxiv.org/pdf/1611.07308.pdf)

1. **Mgae: Marginalized graph autoencoder for graph clustering** *Chun Wang, Shirui Pan, Guodong Long, Xingquan Zhu, Jing Jiang.* CIKM 2017. [paper](https://shiruipan.github.io/pdf/CIKM-17-Wang.pdf)


1. **Link Prediction Based on Graph Neural Networks.** *Muhan Zhang, Yixin Chen.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1802.09691.pdf)

1. **SpectralNet: Spectral Clustering using Deep Neural Networks** *Uri Shaham, Kelly Stanton, Henry Li, Boaz Nadler, Ronen Basri, Yuval Kluger.* ICLR 2018. [paper](https://arxiv.org/pdf/1801.01587.pdf)


1. **Deep Recursive Network Embedding with Regular Equivalence.**
*Ke Tu, Peng Cui, Xiao Wang, Philip S. Yu, Wenwu Zhu.* KDD 2018. [paper](http://cuip.thumedialab.com/papers/NE-RegularEquivalence.pdf)

1. **Learning Deep Network Representations with Adversarially Regularized Autoencoders.**
*Wenchao Yu, Cheng Zheng, Wei Cheng, Charu Aggarwal, Dongjin Song, Bo Zong, Haifeng Chen, Wei Wang.* KDD 2018. [paper](http://www.cs.ucsb.edu/~bzong/doc/kdd-18.pdf)

1. **Adversarially Regularized Graph Autoencoder for Graph Embedding.**
*Shirui Pan, Ruiqi Hu, Guodong Long, Jing Jiang, Lina Yao, Chengqi Zhang.* IJCAI 2018. [paper](https://www.ijcai.org/proceedings/2018/0362.pdf)

1. **Deep graph infomax.** *Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm.* ICLR 2019. [paper](https://arxiv.org/abs/1809.10341)


### Graph Generation

1. **Learning graphical state transitions.** *Daniel D. Johnson.* ICLR 2016. [paper](https://openreview.net/pdf?id=HJ0NvFzxl)

1. **MolGAN: An implicit generative model for small molecular graphs.**
*Nicola De Cao, Thomas Kipf.* 2018. [paper](https://arxiv.org/pdf/1805.11973.pdf)

1. **Learning deep generative models of graphs.** *Yujia Li, Oriol Vinyals, Chris Dyer, Razvan Pascanu, Peter Battaglia.* ICML 2018. [paper](https://arxiv.org/abs/1803.03324)

1. **Netgan: Generating graphs via random walks.** *Aleksandar Bojchevski, Oleksandr Shchur, Daniel Zügner, Stephan Günnemann.* ICML 2018. [paper](https://arxiv.org/pdf/1803.00816.pdf)

1. **Graphrnn: A deep generative model for graphs.** *Jiaxuan You, Rex Ying, Xiang Ren, William L. Hamilton, Jure Leskovec.* ICML 2018. [paper](https://arxiv.org/pdf/1802.08773.pdf)

1. **Constrained Generation of Semantically Valid Graphs via Regularizing Variational Autoencoders.** *Tengfei Ma, Jie Chen, Cao Xiao.* NeurIPS 2018. [paper](https://papers.nips.cc/paper/7942-constrained-generation-of-semantically-valid-graphs-via-regularizing-variational-autoencoders.pdf)

1. **Graph convolutional policy network for goal-directed molecular graph generation.** *Jiaxuan You, Bowen Liu, Rex Ying, Vijay Pande, Jure Leskovec.* NeurIPS 2018. [paper](https://arxiv.org/abs/1806.02473)

1. **D-VAE: A Variational Autoencoder for Directed Acyclic Graphs.** *Muhan Zhang, Shali Jiang, Zhicheng Cui, Roman Garnett, Yixin Chen.* NeuralIPS 2019. [paper](https://arxiv.org/abs/1904.11088)


<a name="stgnn" />

## Spatial-Temporal Graph Neural Networks

1. **Structured sequence modeling with graph convolutional recurrent networks.** *Youngjoo Seo, Michaël Defferrard, Pierre Vandergheynst, Xavier Bresson.* 2016. [paper](https://arxiv.org/pdf/1612.07659.pdf)

1. **Structural-rnn: Deep learning on spatio-temporal graphs.** *Ashesh Jain, Amir R. Zamir, Silvio Savarese, Ashutosh Saxena.* CVPR 2016. [paper](https://arxiv.org/abs/1511.05298)


1. **Know-Evolve: Deep Temporal Reasoning for Dynamic Knowledge Graphs.** *
Rakshit Trivedi, Hanjun Dai, Yichen Wang, Le Song.* ICML 2017 [paper](https://arxiv.org/pdf/1705.05742.pdf)

1. **Deep multi-view spatial-temporal network for taxi.** *Huaxiu Yao, Fei Wu, Jintao Ke, Xianfeng Tang, Yitian Jia, Siyu Lu, Pinghua Gong, Jieping Ye, Zhenhui Li.* AAAI 2018. [paper](https://arxiv.org/abs/1802.08714)

1. **Spatial temporal graph convolutional networks for skeleton-based action recognition.** *Sijie Yan, Yuanjun Xiong, Dahua Lin.* AAAI 2018. [paper](https://arxiv.org/abs/1801.07455)


1. **Diffusion convolutional recurrent neural network: Data-driven traffic forecasting.** *Yaguang Li, Rose Yu, Cyrus Shahabi, Yan Liu.* ICLR 2018. [paper](https://arxiv.org/pdf/1707.01926.pdf)

1. **Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting.** *Bing Yu, Haoteng Yin, Zhanxing Zhu.* IJCAI 2018. [paper](https://arxiv.org/pdf/1709.04875.pdf)

1. **Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting.** *Shengnan Guo, Youfang Lin, Ning Feng, Chao Song, HuaiyuWan* AAAI 2019. [paper](https://aaai.org/ojs/index.php/AAAI/article/view/3881)

1. **Spatio-temporal Multi-Graph Convolution Network for Ride-hailing Demand Forecasting.** *Xu Geng, Yaguang Li, Leye Wang, Lingyu Zhang, Qiang Yang, Jieping Ye, Yan Liu.* AAAI 2019. [paper](http://www-scf.usc.edu/~yaguang/papers/aaai19_multi_graph_convolution.pdf)

1. **Spatio-Temporal Graph Routing for Skeleton-based Action Recognition.** *Bin Li, Xi Li, Zhongfei Zhang, Fei Wu.*  AAAI 2019. [paper](https://www.aaai.org/Papers/AAAI/2019/AAAI-LiBin.6992.pdf)

1. **Graph wavenet for deep spatial-temporal graph modeling** *Z. Wu, S. Pan, G. Long, J. Jiang, and C. Zhang* IJCAI 2019. [paper](https://arxiv.org/abs/1906.00121)

1. **Semi-Supervised Hierarchical Recurrent Graph Neural Network for City-Wide Parking Availability Prediction.** *Weijia Zhang, Hao Liu, Yanchi Liu, Jingbo Zhou, Hui Xiong.* AAAI 2020. [paper](https://arxiv.org/pdf/1911.10516.pdf)
<a name="application" />

## Application

<a name="cv" />

### Computer Vision
1. **3d graph neural networks for rgbd semantic segmentation.** *Xiaojuan Qi, Renjie Liao, Jiaya Jia†, Sanja Fidler, Raquel Urtasun.* CVPR 2017. [paper](http://openaccess.thecvf.com/content_ICCV_2017/papers/Qi_3D_Graph_Neural_ICCV_2017_paper.pdf)

1. **Syncspeccnn: Synchronized spectral cnn for 3d shape segmentation.** *Li Yi, Hao Su, Xingwen Guo, Leonidas Guibas.* CVPR 2017. [paper](https://arxiv.org/pdf/1612.00606.pdf)

1. **A simple neural network module for relational reasoning.** *Adam Santoro, David Raposo, David G.T. Barrett, Mateusz Malinowski, Razvan Pascanu, Peter Battaglia, Timothy Lillicrap.* NIPS 2017 [paper](https://arxiv.org/pdf/1706.01427.pdf)

1. **Situation Recognition with Graph Neural Networks.** *Ruiyu Li, Makarand Tapaswi, Renjie Liao, Jiaya Jia, Raquel Urtasun, Sanja Fidler.* ICCV 2017. [paper](https://arxiv.org/pdf/1708.04320)

1. **Image generation from scene graphs.** *Justin Johnson, Agrim Gupta, Li Fei-Fei.* CVPR 2018. [paper](https://arxiv.org/pdf/1804.01622.pdf)

1. **PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation.**
*Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas.* CVPR 2018. [paper](https://arxiv.org/pdf/1612.00593.pdf)

1. **Iterative visual reasoning beyond convolutions.** *Xinlei Chen, Li-Jia Li, Li Fei-Fei, Abhinav Gupta.* CVPR 2018. [paper](https://arxiv.org/pdf/1803.11189.pdf)

1. **Large-scale point cloud semantic segmentation with superpoint graphs.** *Loic Landrieu, Martin Simonovsky.* CVPR 2018. [paper](https://arxiv.org/pdf/1711.09869.pdf)


1. **Learning Conditioned Graph Structures for Interpretable Visual Question Answering.**
*Will Norcliffe-Brown, Efstathios Vafeias, Sarah Parisot.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1806.07243)

1. **Out of the box: Reasoning with graph convolution nets for factual visual question answering.** *Medhini Narasimhan, Svetlana Lazebnik, Alexander G. Schwing.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1811.00538.pdf)

1. **Symbolic Graph Reasoning Meets Convolutions.** *Xiaodan Liang, Zhiting Hu, Hao Zhang, Liang Lin, Eric P. Xing.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7456-symbolic-graph-reasoning-meets-convolutions.pdf)

1. **Few-shot learning with graph neural networks.** *Victor Garcia, Joan Bruna.* ICLR 2018. [paper](https://arxiv.org/abs/1711.04043)

1. **Factorizable net: an efficient subgraph-based framework for scene graph generation.** *Yikang Li, Wanli Ouyang, Bolei Zhou, Jianping Shi, Chao Zhang, Xiaogang Wang.* ECCV 2018. [paper](https://arxiv.org/abs/1806.11538)

1. **Graph r-cnn for scene graph generation.** *Jianwei Yang, Jiasen Lu, Stefan Lee, Dhruv Batra, Devi Parikh.* ECCV 2018. [paper](https://arxiv.org/pdf/1808.00191.pdf)

1. **Learning Human-Object Interactions by Graph Parsing Neural Networks.** *Siyuan Qi, Wenguan Wang, Baoxiong Jia, Jianbing Shen, Song-Chun Zhu.* ECCV 2018. [paper](https://arxiv.org/pdf/1808.07962.pdf)

1. **Neural graph matching networks for fewshot 3d action recognition.** *Michelle Guo, Edward Chou, De-An Huang, Shuran Song, Serena Yeung, Li Fei-Fei* ECCV 2018. [paper](http://openaccess.thecvf.com/content_ECCV_2018/papers/Michelle_Guo_Neural_Graph_Matching_ECCV_2018_paper.pdf)

1. **Rgcnn: Regularized graph cnn for point cloud segmentation.** *Gusi Te, Wei Hu, Zongming Guo, Amin Zheng.* 2018. [paper](https://arxiv.org/pdf/1806.02952.pdf)

1. **Dynamic graph cnn for learning on point clouds.** *Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon.* 2018. [paper](https://arxiv.org/pdf/1801.07829.pdf)

<a name="nlp" />

### Natural Language Processing
1. **Encoding Sentences with Graph Convolutional Networks for Semantic Role Labeling.**
*Diego Marcheggiani, Ivan Titov.* EMNLP 2017. [paper](https://arxiv.org/abs/1703.04826)

1. **Graph Convolutional Encoders for Syntax-aware Neural Machine Translation.**
*Joost Bastings, Ivan Titov, Wilker Aziz, Diego Marcheggiani, Khalil Sima'an.* EMNLP 2017. [paper](https://arxiv.org/pdf/1704.04675)



1. **Diffusion maps for textual network embedding.** *Xinyuan Zhang, Yitong Li, Dinghan Shen, Lawrence Carin.* NeurIPS 2018. [paper](https://arxiv.org/pdf/1805.09906.pdf)

1. **A Graph-to-Sequence Model for AMR-to-Text Generation.**
*Linfeng Song, Yue Zhang, Zhiguo Wang, Daniel Gildea.* ACL 2018. [paper](https://arxiv.org/abs/1805.02473)

1. **Graph-to-Sequence Learning using Gated Graph Neural Networks.** *Daniel Beck, Gholamreza Haffari, Trevor Cohn.* ACL 2018. [paper](https://arxiv.org/pdf/1806.09835.pdf)




1. **Cross-lingual Knowledge Graph Alignment via Graph Convolutional Networks.** *Zhichun Wang, Qingsong Lv, Xiaohan Lan, Yu Zhang.* EMNLP 2018. [paper](http://www.aclweb.org/anthology/D18-1032)

1. **Graph Convolution over Pruned Dependency Trees Improves Relation Extraction.**  *Yuhao Zhang, Peng Qi, Christopher D. Manning.* EMNLP 2018. [paper](https://arxiv.org/pdf/1809.10185)

1. **Multiple Events Extraction via Attention-based Graph Information Aggregation.** *Xiao Liu, Zhunchen Luo, Heyan Huang.* EMNLP 2018. [paper](https://arxiv.org/pdf/1809.09078.pdf)

1. **Exploiting Semantics in Neural Machine Translation with Graph Convolutional Networks.** *Diego Marcheggiani, Joost Bastings, Ivan Titov.* NAACL 2018. [paper](http://www.aclweb.org/anthology/N18-2078)

1. **Graph Convolutional Networks for Text Classification.** *Liang Yao, Chengsheng Mao, Yuan Luo.* AAAI 2019. [paper](https://arxiv.org/pdf/1809.05679.pdf)
<a name="web" />

### Internet
1. **Graph Convolutional Networks with Argument-Aware Pooling for Event Detection.**
*Thien Huu Nguyen, Ralph Grishman.* AAAI 2018. [paper](http://ix.cs.uoregon.edu/~thien/pubs/graphConv.pdf)

1. **Semi-supervised User Geolocation via Graph Convolutional Networks.**
*Afshin Rahimi, Trevor Cohn, Timothy Baldwin.* ACL 2018. [paper](https://arxiv.org/pdf/1804.08049.pdf)

1. **Adversarial attacks on neural networks for graph data.** *Daniel Zügner, Amir Akbarnejad, Stephan Günnemann.* KDD 2018. [paper](https://arxiv.org/pdf/1805.07984.pdf)

1. **Deepinf: Social influence prediction with deep learning.** *Jiezhong Qiu, Jian Tang, Hao Ma, Yuxiao Dong, Kuansan Wang, Jie Tang.* KDD 2018. [paper](https://arxiv.org/pdf/1807.05560.pdf)

<a name="rec" />

### Recommender Systems. 
1. **Geometric Matrix Completion with Recurrent Multi-Graph Neural Networks.**
*Federico Monti, Michael M. Bronstein, Xavier Bresson.* NIPS 2017. [paper](https://arxiv.org/abs/1704.06803)

1. **Graph Convolutional Matrix Completion.**
*Rianne van den Berg, Thomas N. Kipf, Max Welling.* 2017. [paper](https://arxiv.org/abs/1706.02263)

1. **Graph Convolutional Neural Networks for Web-Scale Recommender Systems.**
*Rex Ying, Ruining He, Kaifeng Chen, Pong Eksombatchai, William L. Hamilton, Jure Leskovec.* KDD 2018. [paper](https://arxiv.org/pdf/1806.01973.pdf)

1. **Session-based Recommendation with Graph Neural Networks.** *Shu Wu, Yuyuan Tang, Yanqiao Zhu, Liang Wang, Xing Xie, Tieniu Tan.* AAAI 2019. [paper](https://arxiv.org/pdf/1811.00855.pdf)

<a name="health" />

### Healthcare
1. **Gram:graph-based attention model for healthcare representation learning** *Edward Choi, Mohammad Taha Bahadori, Le Song, Walter F. Stewart, Jimeng Sun.* KDD 2017. [paper](https://arxiv.org/pdf/1611.07012.pdf)

1. **MILE: A Multi-Level Framework for Scalable Graph Embedding.**
*Jiongqian Liang, Saket Gurukar, Srinivasan Parthasarathy.* [paper](https://arxiv.org/pdf/1802.09612.pdf) 

1. **Hybrid Approach of Relation Network and Localized Graph Convolutional Filtering for Breast Cancer Subtype Classification.** *Sungmin Rhee, Seokjun Seo, Sun Kim.* IJCAI 2018. [paper](https://arxiv.org/abs/1711.05859)

1. **GAMENet: Graph Augmented MEmory Networks for Recommending Medication Combination.** *Junyuan Shang, Cao Xiao, Tengfei Ma, Hongyan Li, Jimeng Sun.* AAAI 2019. [paper](https://arxiv.org/pdf/1809.01852.pdf)
<a name="chemistry" />

### Chemistry
1. **Molecular Graph Convolutions: Moving Beyond Fingerprints.**
*Steven Kearnes, Kevin McCloskey, Marc Berndl, Vijay Pande, Patrick Riley.* Journal of computer-aided molecular design 2016. [paper](https://arxiv.org/pdf/1603.00856.pdf)

1. **Protein interface prediction using graph convolutional networks.** *Alex Fout, Jonathon Byrd, Basir Shariat, Asa Ben-Hur.* NIPS 2017. [paper](https://papers.nips.cc/paper/7231-protein-interface-prediction-using-graph-convolutional-networks.pdf)


1. **Modeling polypharmacy side effects with graph convolutional networks.** *Marinka Zitnik, Monica Agrawal, Jure Leskovec.* ISMB 2018. [paper](https://arxiv.org/abs/1802.00543)


<a name="physics" />

### Physics

1. **Interaction Networks for Learning about Objects, Relations and Physics.**
*Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Rezende, Koray Kavukcuoglu.* NIPS 2016. [paper](https://arxiv.org/pdf/1612.00222.pdf)

1. **Vain: Attentional multi-agent predictive modeling.** *Yedid Hoshen.* NIPS 2017 [paper](https://arxiv.org/pdf/1706.06122.pdf)

<a name="others" />

### Others
1. **Learning to represent programs with graphs.** *Miltiadis Allamanis, Marc Brockschmidt, Mahmoud Khademi.* ICLR 2017. [paper](https://arxiv.org/pdf/1711.00740.pdf)

1. **Combinatorial Optimization with Graph Convolutional Networks and Guided Tree Search.**
*Zhuwen Li, Qifeng Chen, Vladlen Koltun.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7335-combinatorial-optimization-with-graph-convolutional-networks-and-guided-tree-search.pdf)

1. **Approximation Ratios of Graph Neural Networks for Combinatorial Problems.** *Ryoma Sato， Makoto Yamada， Hisashi Kashima.* NeurIPS 2019. [paper](https://arxiv.org/abs/1905.10261
)

1. **Recurrent Relational Networks.**
*Rasmus Palm, Ulrich Paquet, Ole Winther.* NeurIPS 2018. [paper](http://papers.nips.cc/paper/7597-recurrent-relational-networks.pdf)


1. **End to end learning and optimization on graphs.** *Bryan Wilder, Eric Ewing, Bistra Dilkina, Milind Tambe.*  NeurIPS 2019. [paper](https://arxiv.org/abs/1905.13732)

1. **NerveNet: Learning Structured Policy with Graph Neural Networks.** *Tingwu Wang, Renjie Liao, Jimmy Ba, Sanja Fidler.* ICLR 2018. [paper](https://openreview.net/pdf?id=S1sqHMZCb)
<a name="library" />


1. **Graph Neural Network for Music Score Data and Modeling Expressive Piano Performance.** *Dasaem Jeong, Taegyun Kwon, Yoojin Kim, Juhan Nam.* [paper]()


1. **Circuit-GNN: Graph Neural Networks for Distributed Circuit Design.** *GUO ZHANG, Hao He, Dina Katabi* [paper](https://icml.cc/Conferences/2019/Schedule?showEvent=4826)

1. **Alleviating the Inconsistency Problem of Applying Graph Neural Network to Fraud Detection** *Zhiwei Liu, Yingtong Dou, Philip S. Yu, Yutong Deng, Hao Peng.* 2020. [paper](https://arxiv.org/pdf/2005.00625.pdf)

## Library

1. **PyTorch Geometric (PyG)**: [Github](https://github.com/rusty1s/pytorch_geometric) | [Doc](https://pytorch-geometric.readthedocs.io/) | [Examples](https://github.com/rusty1s/pytorch_geometric/tree/master/examples)

1. **Deep Graph Library (DGL)**: [Github](https://github.com/dmlc/dgl) | [Doc](https://docs.dgl.ai/) | [Examples](https://github.com/dmlc/dgl/blob/master/examples/README.md)

1. **tf_geometric**: [Github](https://github.com/CrawlScript/tf_geometric) | [Doc](https://tf-geometric.readthedocs.io) | [Examples](https://github.com/CrawlScript/tf_geometric/tree/master/demo)

1. **Graph Nets library**: [Github](https://github.com/deepmind/graph_nets) | [Doc](https://github.com/deepmind/graph_nets/tree/master/docs)

1. **GNN-based Fraud Detection Toolbox**: [Github](https://github.com/safe-graph/DGFraud)
