MUNIT: Few-Shot Unsupervised Image-to-Image Translation
=======================================================

# Abstract
----------

-	기존 CycleGAN으로 대변되는 Unpaired Image-to-Image translation을 통해 비교적 성공적인 이미지 변환이 있었으나, translation 시에 diverse한 이미지가 나오지 못하는 단점이 존재했다.
-	본 논문에서 제시하는 MUNIT 알고리즘에서는 다음과 같은 차이점이 있다. 
1) 이미지는 content code와 style code로 나누어질 수 있는데, content code는 도메인과 상관없이 공통적인 특질을 가지고 있는 code이고, class code는 도메인 별 달라지는 style code이다. 
2)	따라서, translation을 수행하기 위해서 content code와 style code를 결합해서 이미지를 생성해내는 방법을 제안하고 있다.
3)	또한, 이러한 접근방법은 특정 style image를 제공함으로써 원하는 style로 translation을 control할 수 있는 장점이 있다. 


# Introduction
--------------

-	많은 translation 시나리오는 사실 multi-modal 한 특성을 가지고 있다. 예를 들어, “겨울” 장면은 날씨, 조명 등에 따라 다양한 이미지가 연출될 수 있다. 하지만, 기존 translation 방법론은 deterministic 또는 unimodal한 translation을 전제로 하고 있기 때문에 변환 가능 결과를 모두 커버할 수 있지 못한다. 
-	이미지의 latent space는 content space와 latent space로 나눠질 수 있는데, 다양한 다른 도메인의 이미지가 content space를 공유하고, style space를 공유하지 않는 다는 가정이 있다.
-	따라서 다양한 style code를 샘플링 함으로써 diverse하고 multimodal한 이미지 translation을 수행할 수가 있고, 실제 실험결과 좋은 성능을 보여주는 것으로 본 논문에서 확인이 되었다.
-	한편, style과 content가 분리됨으로써, example-guided image translation (즉 translation target domain의 이미지에 따라 style의 변화)를 수행하게 된다.



# Related Work
--------------

#### Image-to-image translation
-	기존 Image to Image translation의 경우, Translation의 결과가 diverse한 결과를 가질 수 없는 단점이 있다. 이를 극복하기 위해 BicycleGAN 같은 특별한 형태의 GAN이 제안되었으나, 데이터셋이 unpair가 아닌 pair한 이미지가 되어야만 하는 단점이 존재한다. 
-	Diverse한 결과라는 건 mode의 수로 얘기될 수 있는데, mode의 수는 사전에 정해지는 것이 아니기 때문에 어려움이 존재한다.
#### Style transfer
-	일반적인 style transfer의 경우는 single example에서 style을 transfer하지만, Image to image translation의 경우 여러 이미지에서 style을 추출해서 transfer하는 특징이 있다. 본 논문에서 제시하는 알고리즘은 두 경우를 모두 커버가 가능하다.
#### Learning disentangled representations
-	InfoGAN이나 beta-VAE에서 disentangled representation을 시도하였으며, 본 논문에서는 기본이 되는 feature style과 content를 disentangle를 수행했다.


# MUNIT assumptions & modeling
-----------------------------------------
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/2.jpg.png)

각 도메인에서 추출한 이미지 샘플:   일 때, 우리의 목적은 conditional distribution  및  를 찾는 것이며, 이는 
 및  와 같은 translation model를 찾는 것이다. 이러한 model은 multi-modal distribution이며, 기존의 deterministic model로는 표현할 수 없다. 

문제를 풀기 위해서 partially shared latent space assumption를 생각해야 한다. 즉, content latent code와 style latent code를 분리하는데, content code는 두 도메인 모두 share하는 형태이며, style code는 특정 도메인에만 속하는 형태인 것이다.
예를 들어 content code와 style code를 가지고 와서 generator  를 통해 이미지를 생성하는 경우로 생각 할 수 있다.  
 주의할 점은, encoder와 decoder 모두 deterministic하지만,  는 continuous 한 특징이 있다는 것이다. 

본 논문에서 제안하는 코드는 다음과 같다. 
첫 번째 이미지에서 추출된 content code   과
두 번째 도메인에 해당하는 style code  를 활용해서
Image translation  를 수행한다. 즉, 비록  은 unimodal distribution 이지만, decoder의 non-linearity 덕분에 생성되는 translation image는 multimodal이 될 수 있다. 



-	FUNIT을 학습하기 위해서는 다양한 class로 이루어진 unpaired 학습 데이터 셋을 구성해야한다. 학습은 기본적으로 mult-class translation model이 된다.
-	한편 translation 시에는 소량의 이미지 (few images)만을 가지고 target class로의 translation을 수행한다. 
-	일반적으로 Generator는 한 개의 이미지를 가지고 translation을 수행하는데, 여기서는 대신 K장의 이미지를 가지고 translation을 수행하는 것이 특징이다. 여기서 y1…yk 이미지는 모두 동일한 class cy에 속하는 이미지 이며, x는 class cx에 속하는 이미지로써, 서로 다른 class가 되어야 한다.
    ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/3.jpg.png)

 
## Few-shot Image Translator

-	Generator는 few-shot image translator라고 부르기로 한다.
-	생성되는 이미지는 class cy에 속하는 이미지가 될 것이지만, 기본적으로 큰 이미지의 특성은 x와 비슷한 형태를 가질 것이다.
-	Generator G는 총 3개의 네트워크로 구성이 된다. 
    ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/4.jpg.png)

1)	Content Encoder Ex
Content image x를 content latent code Zx로 만들어준다. (Zx는 feature map을 의미)
Class-invariant latent 특성을 찾을 수 있다. (eg. Object pose)

2)	Class Decoder Ey
K개의 class image {y1, … , yk}로 임의의 latent vector로 만들고, 이를 평균내어서 최종 class latent code Zy를 구한다.
Class-specific latent 특성을 찾을 수 있다. (eg. Object appearance)

3)	Decoder Fx
여러 개의 Adaptive instance normalization (AdaIN) residual blocks으로 이루어져있다. AdaIN residual block은 AdaIN을 normalization
layer로 쓰는 residual block을 의미한다. Class latent code를 가지고 global look (eg. . Object appearance)를 컨트롤하게 되고, content image code가 local structure (eg. Locations of eyes)를 컨트롤 할 수 있는 특징이 있다. (추가 주석: AdaIN 이란? AdaIN 은 각 샘플의 채널마다 actgivations zero mean 값과 unit variance 값을 가지도록 normalization을 하게 된다. 그리고 사전에 학습한 affine transformation을 활용하여 변환하게 된다. Affine transformation은 기본적으로 spatially invariant하므로 global appearance information을 찾을 수 있는 특성이 있다.)

-	즉, 학습 동안에 class encoder는 source class (학습 데이터 셋에 존재하는 class를 의미함)에서 class specific 한 정보를 추출하며, 인퍼런스 시간에 이러한 추출 능력이 지금까지 보지못한 소량의 데이터에 일반화가 되는 것이다.


## Multi-task Adversarial Discriminator

-	Discriminator D는 여러 개의 adversarial classification task를 동시에 수행하게 된다. 각각의 task는 binary classificiation task (진짜? 가짜 이미지 구별)이다. 만약에 |S|개의 source class가 존재한다면 D는 |S|개의 결과를 생성하게 된다.
    
#### Discriminator update rule 
1) Source class (Cx)의 진짜 이미지에 대해 판정할 경우, 만약 Cx번째 판정 결과가 가짜(false)일 경우 D에 대해 penalize를 하게 된다
2) Source class (Cx)로부터 생성된 translation image에 대해 판정할 경우, 만약 Cx번째 판정결과가 진짜(positive)일 경우 D를 penalize하게 된다. 
3) 다른 Source class (Cx 제외)에 대해 판정할때는 판정결과가 가짜(false)가 되더라도 D에 대해 별도의 penalize를 하지 않는다.   
Generator update rule
4) G를 업데이트 할때는 G가 생성한 Cx class의 translation이미지가 D로부터 가짜로 판정 (false)될 경우 penalize한다. 


##	Learning
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/5.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/6.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/7.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/8.jpg.png)
  
Df는 Discriminator의 마지막 latent feature vector를 의미하며, feature 간의 거리를 계산하여 보다 feature가 유사한 이미지가 생성되도록 하는 역할을 한다. 


# Experiments
-------------

Few-shot 환경에 맞춰, 학습되는 이미지를 K=1,5,10,20으로 변경하가면서 학습을 수행한 결과는 다음과 같다.
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/9.jpg.png)
 

-	본 논문과 비교를 하기 위한 Baseline 모델 및 데이터 셋 환경은 다음과 같다.

1)	Fair dataset
본 논문에서 제안하는 FUNIT의 환경이다. 인퍼런스 시에 이미지는 학습에서 본적이 없는 이미지 이며, Multi-class 부분 SOTA 모델로 알려진 StarGAN 모델을 확장해서 실험하였다. 학습은 source class만 가지고 했으며, 인퍼런스 시에는 각 target class마다 K개의 이미지를 가지고 VGG feature의 평균값을 구하고, 각 source class의 평균 VGG feature 값과 비교한다. 가장 가깝게 나오는 source class를 선정해서 target class에 해당하는 label vector로 선정 후에 StarGAN을 수행하도록 한다.

2)	Unfair
학습 때에 인퍼런스에 쓰이는 target class에 해당하는 이미지도 함께 학습하는 환경이다. StarGAN 같은 경우는 각 source class마다 하나의 도메인이 되는 것이며, CycleGAN 같은 경우는 source class들의 이미지는 모두 첫번째 도메인, 한 개의 특정 target class는 두번째 도메인으로 설정해서 실험을 진행하였다.

-	본 논문에서 사용한 Performance metric은 다음과 같다.

1)	Train accuracy
Translation 된 이미지가 target class에 속하는 것으로 판정이 되는지 측정한 값이다. 즉, translation이 잘되었다면 classifier가 translation 이미지를 target class에 속한다고 생각하고 분류할 것이라는 가정이다.

2)	Content preservation
도메인이 바뀌더라도 domain-invariant 한 특징이 잘 keep되는지 판단하는 지표이다. 도메인 변환에 invariant한 두 VGG conv5 feature들간의 거리로 판단한다.

3)	Photorealism
Inception score로 판단하며, 얼마나 진짜 이미지 같은 양질의 이미지가 생성되었는지를 판단하는 기준이다. (양질의 이미지: 실제 이미지의 분포를 따르는 이미지의 여부)

     ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/10.jpg.png)

 
4)	Distribution matching
두 이미지 데이터셋에 대한 feature 유사도를 측정하는 지표이며, 진짜 데이터 셋과 진짜 데이터로ranslation 데이터 셋간의 feature의 거리를 가지고 측정한다. 진짜 이미지에 있는 feature의 variance가 잘 반영된 이미지가 생성되었는지 볼 수 있다. 

-	본 논문에서 제안하는 FUNIT은 다른 Baseline 모델의 성능을 모두 능가하는 결과를 보이는데, 특히 주목할 만한 것은 5-shot으로 수행한 FUNIT 모델의 경우 인퍼런스 target class 이미지를 학습 중에 본적이 없음에도 불구하고 그 성능이 unfair 환경의 CycleGAN보다 좋게 나오며, 특히 단일 Generator로 나온 성능이라는 점이 주목할 만하다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/11.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/12.jpg.png)
 
 
-	육안으로 볼 수 있는 성능 역시, FUNIT이 좀더 좋은 성능을 보인다는 것을 알 수 있다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/13.jpg.png)
 
-	한편, source class수가 증가하면 FUNIT 모델의 성능이 좋아지는 것을 볼 수 있는데, 학습 중에 보다 많은 class diversity를 보면 성능에 도움이 된다는 것을 알 수 있다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/14.jpg.png)
 
 
-	본 알고리즘은 단순히 GAN이 아닌, Few-shot classification 목적으로 쓸 수도 있는데, 수량이 부족한 target class에 대해 translation 된 이미지 (생성이미지: 1, 50, 100) 를 학습데이터에 넣어서 classification을 수행할 경우 그 성능이 기존 유사 접근방식의 논문 보다 좋은 것을 볼 수 있다. 

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/15.jpg.png)
 

-	본 알고리즘의 성능이 제한적으로 나오는 경우는, 기존의 translation 알고리즘이 가지고 있는 단점처럼 shape이 급격하게 변하는 경우 인 것도 확인 할 수 있으며, 추후 연구가 될 분야이다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/16.jpg.png)
 
-	다음은 Animal 및 Face dataset에 대한 수행 결과를 보여주는 예시 이다.
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/17.jpg.png)
 


                                                                                            -  END - 
