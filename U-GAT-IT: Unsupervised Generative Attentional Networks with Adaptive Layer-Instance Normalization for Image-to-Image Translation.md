U-GAT-IT: 
=======================================================

## Introduction
- 기존에 많은 Image to image translation 논문들이 나왔지만, 대부분은 style transfer와 유사하게 texture를 변환하는 작업에 초점이 맞춰져 있고, shape을 변환시키는 작업에는 약점을 드러 냄 
  (예시: photo2vangogh / cat2dog)
- 본 논문에거는 attention module고 새로운 normalization 방법론을 적용하여 end-to-end로 학습이 가능한 방법론을 제시 함
- 본 논문의 가장 큰 특징은 다음과 같음
   1) 새로 추가되는 aux classifier를 통해 얻어지는 attention map을 활용하여 source와 target 도메인 간 translation에 있어서 중요한 부분과 중요하지 않은 부분을 구분해 집중할 수 있는 장점.
      attention map이 generator와 discriminator에 모두 반영되어 단순히 texture가 아닌 shape translation 까지 이루어질 수 있는 구조 임
      - generator : 이미지 생성 시 두 도메인 간의 차이에 집중 (어떤 부분에 집중해면 더욱 진짜 처럼 생성이 되는지?) 
      - discriminator : 진짜 이미지와 생성 이미지 간의 차이 집중 (어떤 부분에 집중하면 진짜/가짜 구분이 잘 되는지?)
   2) Adaptive Layer Instance Normalization (AdaLIN) 을 도입 함. 이는 기존에 Instance Normalization과 Layer Normalization을 adaptive하게 선택해서 학습하는 방법 임
      AdaLIN 덕분에 모델이 shape과 texture의 변화 정도를 좀 더 쉽게 컨트롤 할 수 있는 역할을 하게 되며, 이로 인해 모델 구조를 바꾸거나 할 필요가 없어짐



## 2.1 Model

$$E_s$$

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)

$$\eta$$ 

_s(x)
$$f(x)= if x < x_{min} : (x/x_{min})^a$$ 



U-GAT-IT: 
============================

## Introduction

- 기존에 많은 Image to image translation 논문들이 나왔지만, 대부분은 style transfer와 유사하게 texture를 변환하는 작업에 초점이 맞춰져 있고, shape을 변환시키는 작업에는 약점을 드러 냄 
  (예시: photo2vangogh / cat2dog)
- 본 논문에거는 attention module고 새로운 normalization 방법론을 적용하여 end-to-end로 학습이 가능한 방법론을 제시 함
- 본 논문의 가장 큰 특징은 다음과 같음
   1) 새로 추가되는 aux classifier를 통해 얻어지는 attention map을 활용하여 source와 target 도메인 간 translation에 있어서 중요한 부분과 중요하지 않은 부분을 구분해 집중할 수 있는 장점.
      attention map이 generator와 discriminator에 모두 반영되어 단순히 texture가 아닌 shape translation 까지 이루어질 수 있는 구조 임
      - generator : 이미지 생성 시 두 도메인 간의 차이에 집중 (어떤 부분에 집중해면 더욱 진짜 처럼 생성이 되는지?) 
      - discriminator : 진짜 이미지와 생성 이미지 간의 차이 집중 (어떤 부분에 집중하면 진짜/가짜 구분이 잘 되는지?)
   2) Adaptive Layer Instance Normalization (AdaLIN) 을 도입 함. 이는 기존에 Instance Normalization과 Layer Normalization을 adaptive하게 선택해서 학습하는 방법 임
      AdaLIN 덕분에 모델이 shape과 texture의 변화 정도를 좀 더 쉽게 컨트롤 할 수 있는 역할을 하게 되며, 이로 인해 모델 구조를 바꾸거나 할 필요가 없어짐



## 2.1 Model


![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)

generator는 encoder $$E_s$$, $$decoder G_t$$, $$aux classifier  \eta_s(x)$$ 로 이루어져 있으며, 여기서  aux classifier는 input 이미지가 특정 도메인 $$X_s$$ 에서 왔는지에 대한 확률을 표현하며, 다음과 같이 encoder의 각 k-th feature map이 global average pooling 및 global max pooling을 된 후 곱해는 weight $$W^k_s$$를 학습한다.
$$\eta_s(x) = \sigma(\sum_kw^k_s\sigma_(ij)E^{k_(ij)}(x))$$ 

이를 통해서 domain specific attention feature map $$a_s(x) = w_s * E_s(x) = w^k_s * $$  



## Related Work
--------------
-   기존 모델의 단점으로는,  
    1)	Sample inefficient: 학습 시에 class (또는 domain) 별 다량의 이미지가 필요하다.
    2)	모델을 만들더라도, 그 모델은 특정 두 class간의 translation에만 사용이 가능하다.
      (내 생각: 이건 one generator 모델인 StarGAN 같은 모델은 해당 안됨)
-	본 논문과 유사한 few-shot쪽으로 GAN을 적용한 기존 논문이 있는데 (One-shot unsupervised cross domain translation. In Advances in Neural Information Processing Systems (NIPS), 2018) 기존 논문은 학습 시에는 class마다 한 장의 이미지만 가지고 있다고 가정하고, translation 하고자 하는 target class에는 많은 이미지가 있다고 가정하고 있는 반면, 본 논문은 반대로 학습 시에는 많은 이미지가 있고, translation 시에만 확보할 수 있는 소량의 이미지가 있다고 가정하고 있다.
-	본 논문과 유사한 multi class 쪽으로 접근한 논문도 있는데, 이 논문들은 translation 대상인 class가 학습 데이터셋 에도 있다는 가정을 하고 있다.   
-	한편, 기존에 많은 few-shot 관련 논문이 존재하나, few-shot translation을 시도한 것은 이 논문이 최초이라고 볼 수 있다.

## 	Few-shot Unsupervised Image Translation
-----------------------------------------
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/1.jpg)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/2.jpg.png)

-	FUNIT을 학습하기 위해서는 다양한 class로 이루어진 unpaired 학습 데이터 셋을 구성해야한다. 학습은 기본적으로 mult-class translation model이 된다.
-	한편 translation 시에는 소량의 이미지 (few images)만을 가지고 target class로의 translation을 수행한다. 
-	일반적으로 Generator는 한 개의 이미지를 가지고 translation을 수행하는데, 여기서는 대신 K장의 이미지를 가지고 translation을 수행하는 것이 특징이다. 여기서 y1…yk 이미지는 모두 동일한 class cy에 속하는 이미지 이며, x는 class cx에 속하는 이미지로써, 서로 다른 class가 되어야 한다.
    ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/3.jpg.png)
 
#### Few-shot Image Translator

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


#### Multi-task Adversarial Discriminator

-	Discriminator D는 여러 개의 adversarial classification task를 동시에 수행하게 된다. 각각의 task는 binary classificiation task (진짜? 가짜 이미지 구별)이다. 만약에 |S|개의 source class가 존재한다면 D는 |S|개의 결과를 생성하게 된다.
    
###### Discriminator update rule 
1) Source class (Cx)의 진짜 이미지에 대해 판정할 경우, 만약 Cx번째 판정 결과가 가짜(false)일 경우 D에 대해 penalize를 하게 된다
2) Source class (Cx)로부터 생성된 translation image에 대해 판정할 경우, 만약 Cx번째 판정결과가 진짜(positive)일 경우 D를 penalize하게 된다. 
3) 다른 Source class (Cx 제외)에 대해 판정할때는 판정결과가 가짜(false)가 되더라도 D에 대해 별도의 penalize를 하지 않는다.   
Generator update rule
4) G를 업데이트 할때는 G가 생성한 Cx class의 translation이미지가 D로부터 가짜로 판정 (false)될 경우 penalize한다. 


#### Learning
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/5.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/6.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/7.jpg.png)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/funit/8.jpg.png)
  
Df는 Discriminator의 마지막 latent feature vector를 의미하며, feature 간의 거리를 계산하여 보다 feature가 유사한 이미지가 생성되도록 하는 역할을 한다. 


## Experiments
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
