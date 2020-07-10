

# MUNIT:  Multimodal Unsupervised Image-to-image Translation


### Abstract
-	기존 CycleGAN으로 대변되는 Unpaired Image-to-Image translation을 통해 비교적 성공적인 이미지 변환이 있었으나, translation 시에 diverse한 이미지가 나오지 못하는 단점이 존재했다.
-	본 논문에서 제시하는 MUNIT 알고리즘에서는 다음과 같은 차이점이 있다. 

    1)	이미지는 content code와 style code로 나누어질 수 있는데, content code는 도메인과 상관없이 공통적인 특질을 가지고 있는 code이고, class code는 도메인 별 달라지는 style code이다. 
    2)	따라서, translation을 수행하기 위해서 content code와 style code를 결합해서 이미지를 생성해내는 방법을 제안하고 있다.
    3)	또한, 이러한 접근방법은 특정 style image를 제공함으로써 원하는 style로 translation을 control할 수 있는 장점이 있다. 
    
    
    
    
### Introduction
-	많은 translation 시나리오는 사실 multi-modal 한 특성을 가지고 있다. 예를 들어, “겨울” 장면은 날씨, 조명 등에 따라 다양한 이미지가 연출될 수 있다. 하지만, 기존 translation 방법론은 deterministic 또는 unimodal한 translation을 전제로 하고 있기 때문에 변환 가능 결과를 모두 커버할 수 있지 못한다. 

-	이미지의 latent space는 content space와 latent space로 나눠질 수 있는데, 다양한 다른 도메인의 이미지가 content space를 공유하고, style space를 공유하지 않는 다는 가정이 있다.

-	따라서 다양한 style code를 샘플링 함으로써 diverse하고 multimodal한 이미지 translation을 수행할 수가 있고, 실제 실험결과 좋은 성능을 보여주는 것으로 본 논문에서 확인이 되었다.

-	한편, style과 content가 분리됨으로써, example-guided image translation (즉 translation target domain의 이미지에 따라 style의 변화)를 수행하게 된다.




### Related Work

###### Image-to-image translation

-	기존 Image to Image translation의 경우, Translation의 결과가 diverse한 결과를 가질 수 없는 단점이 있다. 이를 극복하기 위해 BicycleGAN 같은 특별한 형태의 GAN이 제안되었으나, 데이터셋이 unpair가 아닌 pair한 이미지가 되어야만 하는 단점이 존재한다. 
-	Diverse한 결과라는 건 mode의 수로 얘기될 수 있는데, mode의 수는 사전에 정해지는 것이 아니기 때문에 어려움이 존재한다.

###### Style transfer
-	일반적인 style transfer의 경우는 single example에서 style을 transfer하지만, Image to image translation의 경우 여러 이미지에서 style을 추출해서 transfer하는 특징이 있다. 본 논문에서 제시하는 알고리즘은 두 경우를 모두 커버가 가능하다.
Learning disentangled representations
-	InfoGAN이나 beta-VAE에서 disentangled representation을 시도하였으며, 본 논문에서는 기본이 되는 feature style과 content를 disentangle를 수행했다.




### MUNIT 가정 & 모델

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/1.jpg)
각 도메인에서 추출한 이미지 샘플: ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/2.jpg)
일 때, 우리의 목적은 conditional distribution   ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/3.jpg)
  및  ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/4.jpg)
 를 찾는 것이며, 이는 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/5.jpg) 및 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/6.jpg) 와 같은 translation model를 찾는 것이다. 이러한 model은 multi-modal distribution이며, 기존의 deterministic model로는 표현할 수 없다. 

문제를 풀기 위해서 partially shared latent space assumption를 생각해야 한다. 즉, content latent code와 style latent code를 분리하는데, content code는 두 도메인 모두 share하는 형태이며, style code는 특정 도메인에만 속하는 형태인 것이다.

예를 들어 content code와 style code를 가지고 와서 generator ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/8.jpg)
  를 통해 이미지를 생성하는 경우로 생각 할 수 있다. ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/9.jpg)
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/10.jpg)

주의할 점은, encoder와 decoder 모두 deterministic하지만,  ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/11.jpg) 은 continuous 한 특징이 있다는 것이다. 


![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/12.jpg) 

 
본 논문에서 제안하는 코드는 다음과 같다. 

첫 번째 이미지에서 추출된 content code ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/13.jpg)   과
두 번째 도메인에 해당하는 style code ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/14.jpg) 를 활용해서
Image translation ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/15.jpg) 를 수행한다. 즉, 비록 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/16.jpg) 은 unimodal distribution 이지만, decoder의 non-linearity 덕분에 생성되는 translation image는 multimodal이 될 수 있다. 

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/17.jpg)



### Loss
여기서는 MUNIT에 쓰이는 loss 함수를 자세하게 알아보겠다.

###### Bidirectional reconstruction loss
Encoder와 decoder를 학습하기 위해 image → latent → image 와 latent → image → latent 방향의 reconstruction loss를 설계 한다.



Image reconstruction
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/18.jpg)


Latent reconstruction
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/19.jpg)


 
마찬가지로 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/20.jpg) 도 같은 방법으로 계산이 된다.

Lreconsi: 다른 style code로부터 각각 다른 이미지가 생성되도록 제약을 준다. 왜냐하면, 도메인1의 c1과 도메인2의 s2를 가지고 x2를 생성 후, E2s로 도메인 2의 style을 encoding해서 원래 s2와 유사하게 만들게 됨 -> s2가 달라지면 이미지도 달라져야 하기 때문이다.

Lreconci: 생성된 이미지가 content 정보를 잘 보존하고 있도록 제약을 준다. 왜냐하면, 도메인1의 c1과 도메인2의 c2를 가지고 x2를 생성 후, 다시 E2c로 c1을 encoding해서 c1과 유사하게 만들게 되기 때문이다.

###### Adversarial loss
일반적으로 쓰이는 GAN Adversarial loss를 의미한다.
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/21.jpg = 150x150)

 
###### Total loss
위에서 언급한 loss들을 다음과 같이 종합할 수 있다.

![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/22.jpg)

 
### Result
 여기서 주의 깊게 봐야할 것은, MUNIT의 경우 기존 Baseline (CycleGAN)과는 다르게 앞서 설명한 multi-modal 성질이 반영 된다는 것이다. 즉, 신발을 translation 시킬 경우 단순하게 하나의 이미지로 translation 되는 것이 아닌 다양한 색상의 이미지로 translation 되는 것을 확인 할 수 있다. 이는 Baseline 모델과 가장 큰 차이인데, 초반에 기술한대로 "겨울" 이라는 도메인으로 translation 되더라도 다양한(multi-modal) 겨울 이미지가 연출 될 수 있음을 얘기하고 있는 것이다.
 
 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/23.jpg)
 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/24.jpg)
 ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/25.jpg)
 
 이는 이미지 quality 뿐만 아니라 diversity에서도 확인이 되는데, 특히 위에서 언급했던 loss를 일부 적용하는 ablation study에서도 MUNIT의 우수성을 확인 할수가 있다.(이 외에 CIS, IS 스코어 관련 내용은 논문 마지막을 참조하면 된다.)
 
![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/munit/26.jpg)
 
 
   
 
 





