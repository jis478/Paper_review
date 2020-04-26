TraVeLGAN: Image-to-image Translation by Transformation Vector Learning
======================================================================= 

# Abstract

-	기존 논문들의 Unpaired image-to-image translation은 cycle-consistency의 가정하에 많이 시도되었고, 그 결과가 가장 만족스러웠음. 
-	하지만 기존 논문에서 주로 얘기하는 translation은 서로 유사한 도메인 (예를 들어 말 -> 얼룩말, 낮의 거리 -> 밤의 거리) 위주였으며, 도메인의 차이가 주로 스타일이나 텍스쳐에서 발생하는 경우였음.
-	본 논문에서는 이렇게 유사한 도메인이 아닌, 서로 많이 다른 도메인 간의 translation을 타켓으로 하고 있음. 이를 위해서는 기존에 써왔던 cycle-consistency를 계속 고집하면 안된다고 주장 함
-	따라서, Siamese 네트워크로 vector transformation을 배워서 GAN 이미지 생성에 활용하는 기법을 제안 함. 즉, 기존 일반적인 GAN은 generator가 생성하는 이미지가 올바른 방향으로 생성하도록 discriminator를 이용했지만, 본 논문에서는 Siamese 넷을 추가하여 3개의 네트워크로 학습을 진행 함. (To this two-network system we add a third: a siamese network that guides the generator so that each original image shares semantics with its generated version)
-	따라서 더 이상 cycle-consistency 제약이 필요 없기 때문에, 보다 flexible한 이미지 tranlsation이 가능 함

# Introduction
-	Unpaired image to image translation을 하는 경우, 특정 이미지가 translation 된 후에 이미지의 주요한 속성이 아직 남아있다는 보장을 할 수가 없다. 이를 보장하기 위해서 지금까지 여러 기법(제약; regularization)들이 쓰여왔으며, 가장 많이 쓰이는 기법은 cycle-consistency로써 generator가 서로 inverse 관계에 있도록 제약을 주는 방법이다. 
-	일반적인 GAN에서는 latent space를 학습하기 위해서 generator와 discriminator를 병행해서 학습하게 되는데, 본 논문에서는 Siamese network까지 활용해서 latent space를 배우는 것이 특징이며, 이를 통해 cycle-consistency를 제거할 수 있게 된다.
-	즉, generator가 이미지를 생성할 때 latent space에서 두 점 (이미지로부터 각각 매핑 된)의vector arithmetic을 유지하면서 생성하도록 하게 된다. 즉, 동일한 도메인에서의 두 이미지 사이의 transformation vector는 변환된 두 이미지간의 transformation vector와 동일해야 한다는 것이다.  이는 word2vec embedding에서 아이디어를 가져온 것으로, 쉽게 말하면 특정 도메인에 있는 이미지 A에서 사과가 오른쪽 위에 위치한 경우, 이 사과의 위치를 왼쪽 아래로 옮기는 이미지 vector transform을 생각해보면, generator가 생성하는 두 이미지 역시 같은 vector transform으로 생각할 수 있는 것이다. 
-	TraVelGAN에서는 latent space를 학습하는 동시에 이러한 vector transform을 잘 생성하도록 학습을 진행하게 된다.
결론적으로, TraVelGAN은 다음과 같은 특징이 있다.
    1.	Cycle-consistency 등 generator에 대해 부여되는 제약이 없음
    2.	새로운 네트워크 (Siamese)를 도입
    3.	Latent space에 대한 해석력 증대 
-	이러한 특징 덕분에 기존 알고리즘으로 수행할 수 없었던 급격한 shape 변화가 있는 translation을 수행할 수가 있다.
-	특히, generator에 부여되는 cycle-consistency 제약은 inverse가 쉬운 방향으로 generator를 학습하게 만들기 때문에 (inverse가 잘 되어야만 원본->가짜->원본으로 translation 된 이미지와 원본 이미지 간의 loss가 줄어들기 때문임), 실제 translation이 inverse가 매우 어려운 문제라면 inverse를 단순화하기 때문에 낮은 translation 성능을 보일 수 밖에 없다. 또한, cycle-consistency loss는 pixel-wise MSE를 기반으로 계산되기 때문에 MSE를 줄이기 위해 생성되는 이미지가 mean 이미지로 생성 하게끔 유도하게 되는 결점이 존재한다.

# Model
-	Notations: ![Representative image](https://github.com/jis478/Paper_review/blob/master/imgs/travelgan/imgs/1.jpg)
 X 도메인에 속하는 이미지, Y도 같은 방법으로 표기
  Y->X로 translation
-	Domain membership:   
-	Individuality:  사이에 관계가 있어야 함. 즉, xi로부터 translation되는 이미지는 Y 도메인의 아무 이미지가 아닌, xi와 유사한 특정 이미지여야 함. CycleGAN에서는 Cycle-consistency로 제약을 주었지만, 여기서는 xi 이미지들간의 관계 와 GXY(xi) 이미지들의 관계를 암묵적(implicit)하게 매칭시키는 방법으로 제약을 부여 함
-	    이는 vector의 방향 뿐만 아니라 크기(magnitude)까지 모두 같아야 한다는 것을 의미 함
-	하지만, 이미지에 바로 위의 공식을 적용하는 것은 pixel 단위로 관계를 보는 것을 의미하기 때문에, word2vec embedding과 유사하게 feature 간의 관계를 보는 것으로 다시 생각해볼 수 있다.   여기서 S는 high-level semantic feature를 배우는 네트워크를 의미한다.
-	즉, 네트워크 S는 원본 도메인에 있는 이미지들을 latent space상으로 매핑하는데, 매핑된 point들의 관계가 translation한 후의 이미지 (타겟 도메인) 들을 latent space상으로 매핑한 point들의 관계와 동일하게 만드는 것이 목표가 된다. 
 
-	여기서 G와 S는 상호 협력적인 관계를 가지게 되는데 (G와 D는 적대적 관계), 두 네트워크가모두 loss를 낮추는 방향으로 학습이 되기 때문이다. 하지만 하나의 문제가, 만약 위의 loss 만으로 학습을 진행한다면 S는 항상 “0”의 값만 내뱉는 네트워크로 훈련이 될 수 있기 때문에 (0으로 내뱉으면 loss가 0이 됨) 여기에 대한 추가 제약이 필요하다.  여기서 S는 Siamese network이기 때문에, 원래 Siamese 네트워크에서 자주 사용하는 contrastive loss를 응용하게 된다.  

최종 loss는 다음과 같다. 
 
 

# Experiments
Similar domains (유사 도메인간 Translation)
Translation이 잘되는 것을 볼 수 있으나, 정량적인 수치로 표현한다면 사실 CycleGAN이 더 잘 translation하는 것으로 볼 수 있다. 즉, 기본적인 global feature를 공유하는 translation에서는 TravelGAN이 SOTA의 성능을 보이는 것이 아니라는 것을 확인할 수 있다.
 
주된 이유는 Loss 설정에서 기인할 수 있는데, TravelGAN에서는 MSE 기반의 Cycle-consistency loss가 없기 때문에 pixel-wise 보다는 semantic feature 위주로 학습이 되고, 따라서 의도치 않은 translation이 일어나기도 하는 것을 볼 수가 있다. (모자만 제거하고 싶은데 머리 색상 이나 배경 색상 바뀜)
신발 예제에서는 이러한 특성이 더욱 강하게 나타나는데, latent space는 semantic feature를 대변한다는 것을 생각해보았을 때, TravelGAN의 latent space 상에서 계산하는 loss는 pixel-wise와는 다르게 원본 이미지에 국한되지 않고 다양한 색상을 가진 신발로 생성할 수 있도록 하는 것으로 볼 수 있다.
 
 


 














Diverse Domains (이종 도메인 간 translation)
상기 예제에서는 유사한 도메인간의 translation을 보았다면, 이번 예제에서는 서로 완전히 다른 형태의 domain 간의 translation을 테스트 한다. 이는 전통적으로 Image to Image translation에서 취약하게 나타나는 영역이며, TravelGAN을 활용하면 급격한 변화가 일어나는 translation에서도 좋은 결과를 볼 수 있는 것을 확인 할 수 있다.
 
 
특히 위에 그림은 Abacus에서 Crossword로 변환한 예제인데, cycle consistency loss를 쓸 경우에는 overfitting한 translation (검은 돌은 무조건 거의  -> 하얀 영역으로 변환)이 일어나서 실제 Crossword 같은 이미지 생성이 어려운 반면, TravelGAN은 이와 다르게 one-to-many translation이 가능하기 때문에 (검은돌 -> 하안영역, 검은 영역으로 모두 변환 가능) 보다 현실적인 crossword가 생성된 다는 것을 볼 수 있다.
 
또한 상기 이미지처럼, 임의로 crossword의 영역을 바꿔가면서 원본 이미지를 가정해본 경우, transaltion 결과 역시 대응하는 형태로 생성되기 때문에 원본 <-> 타켓 도메인의 semantic feature가 잘 mapping되었음을 알 수 있다. 임의로 영역을 바꾼 이미지는 처음부터 학습 데이터에 안들어 갔다는 것을 생각해보면, 이는 TravelGAN이 feature위주로 학습이 된 것을 방증하며, pixel-wise MSE를 쓰는 Cycle-consistency loss로는 확인하기 힘든 특징이다.

 
한편, pixel-wise MSE를 쓰는 CycleGAN의 경우, 별도의 거리에 대한 제약이 없어도 원본 이미지간의 거리와 생성되는 이미지간의 거리가 높은 상관관계를 가지게 된다. (예: 원본 이미지 A,B 사이의 pixel-wise 거리와 translation 이미지 A’와 B’ 사이의 거리 간 상관 관계) 하지만 이미지 생성 시에 이게 항상 좋은 특징이 아닌 것이, 급격한 translation의 경우 A’과 B’ 모두 거리의 속성은 유지하지만 이상하게 생성되는 경우가 많이 발생하게 된다.

하지만 TravelGAN에서는 예상한대로 latent space에서 그 상관 관계가 높게 나오게 되는 것으로 보아, latent space에서 거리의 속성이 잘 보존되며, feature 단위에서 loss를 계산하는 flexible한 특징 덕분에 생성되는 이미지의 품질도 좋은 것을 알 수 있다.

완전히 다른 도메인이기 때문에 이전의 유사한 도메인 변환을 평가하는 지표와는 다른 FID와 Disrciminator score를 썼으며, TravelGAN이 모두 우수한 성능을 보이고 있다.
 






# 비전검사에 대한 적용 가능성
본 논문은 급격한 형태가 있는 이미지 간의 변환을 타켓으로 하고 있다. 이는 비전검사에서도 적용될 수 있는데, 양품 – 불량 이미지가 큰 관점에서 Global feature를 상호 공유(화면 구도 등) 하고, feature가 많이 다른 경우에 적용가능 할 것이다. 즉, 기존에 CycleGAN이나 MUNIT으로 변환할 수 없었던 큰 변화 (LGC side dent의 큰 dent 영역 변환)에 대해 시도해볼 수 있으며, 기존 CycleGAN 계열의 알고리즘 대비 양품 -> 불량 변환의 실패케이스가 얼마나 줄어드는지 함께 실험이 필요하다. 


  
