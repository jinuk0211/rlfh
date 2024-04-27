강화학습 part 3 - ORPO(odds ratio preference optimization)
요즘 핫함 - 카이스트생이 고안

train 할 때 cost ↓↓ performance ↑↑> PPO,DPO

kaist - 2024년 3월 14일자 논문 
https://arxiv.org/pdf/2403.07691.pdf

https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi?usp=sharing

ORPO는 새로운 미세조정(fine-tuning) 기술로, 전통적인 supervised fine-tuning과 preference alignment 단계를 하나의 과정으로 통합해 수행함

이를 통해 훈련에 필요한 계산 자원과 시간을 줄일 수 있음. 또한 실험 결과에 따르면 ORPO는 다양한 모델 크기와 벤치마크에서 다른 alignment 방법들보다 뛰어난 성능을 보임.(PPO, DPO)

기본적으로 모델 instruction tuning(unseen task에 대한 일반화 성능), preference alignment(인간처럼 행동하게 align)의 두가지 단계 겪음

page 1

ORPO 는 PPO,DPO와 달리 참조 모델을 두지 않고 rejected response(bad)에 약한 페널티를 주고 chosen response(win)에 strong adaption signal을 준다. 또한 SFT과정도 없다

먼저 alignment에 사용하는 pairwise 쌍데이터에서의 SFT의 역할과 영향을 조사한다
-> SFT 중에 이상한(disfavored) generation style (너무 딱딱하다거나 수사학적인 어투 같은) 학습하는 것에 페널티를 부여하는 것으로 preference-aligned SFT를 성립시키기에 충분

![image](https://github.com/jinuk0211/rlfh/assets/150532431/b7e33762-b9d6-475f-8f46-694f9167d312)


Related Work
PPO 설명하고 DPO 설명하고 human feedback이 아닌 RLAIF같은 대안 설명

instability of PPO와 sensitivity of the reward model 같은 challenge가 존재


처음 듣는거 
DPO의 잠재적인 overfitting 문제 -> identity preference optimization (IPO). Ethayarajh et al. (2024)

pairwise 데이터를 필요로 하지 않는 alignment
Kahneman-Tversky Optimisation (KTO)
Unified Language Model Alignment (ULMA)

ORPO
지도 미세조정(supervised fine-tuning)과 선호 정렬을 통합하기 위해 reference response의 softmax 값을 음의 로그 가능성 손실(negative log-likelihood loss)에 포함시키는 것을 설명하고 있다 

SFT 만을 사용해 model을 align 시킬려는 시도가 잇었는데 (fine-grained filetering and curation된 작은 데이터셋) 이 시도가 충분히 가능함을 입증한 바가 있다

자연어 처리 분야에서의 fine-grained:

Fine-grained 감정 분석: 감정을 단순히 긍정/부정으로 구분하지 않고, 행복, 슬픔, 화남 등 더 세분화된 감정 범주로 분류
Fine-grained 명명체 인식: 단순 명명체 인식이 아닌 사람, 조직, 위치 등 세부 유형 분류

어쨋든 결론 : alignment에서 SFT가 상당히 중요한 역할을 한다. but 충분히 연구되지 않음

->>따라서 alignment에서 SFT를 첫번째로 study
SFT의 손실 함수 분석과 훈련된 SFT 모델의 preference comprehension 능력에 대한 실증적인 입증 수행함



위의 SFT 관련
이는 의도치 않게 바람직하지 않은 스타일의 토큰 생성 가능성을 높이게 되는데, 이는 그림 3에서 설명된다. 따라서 SFT의 도메인 적응 역할은 유지하면서도 원치 않는 생성 스타일을 식별하고 완화할 수 있는 방법을 개발할 필요가 있음

Absence of Penalty in Cross-Entropy Loss

SFT는 적절한 토큰의 로그 확률을 높임으로써 pretrained 언어 모델을 원하는 도메인에 맞추는 데 중요한 역할을 한다(Zhou et al., 2023a; Dong et al., 2024)
하지만 cross entrophy loss는 참조 answer을 위한 predicted logit이 낮을 때 페널티를 부여하지 non-answer 토큰에는 아무 페널티 or 보상을 주지 않는다.

요약 : chosen response에만 보상을 주지 rejected answer에는 아무런 페널티 메커니즘이 없다 SFT loss <- alignment의 관점에 부적절


![image](https://github.com/jinuk0211/rlfh/assets/150532431/af3453f2-4fee-4790-84b1-2611fae346a3)

![image](https://github.com/jinuk0211/rlfh/assets/150532431/5ba507a1-6bcf-4574-81a1-890b3df53d39)


SFT loss = 참조 answer(토큰)을 생성할 가능성을 최대화
OR loss = 답 y(win)와 바람직하지 않은 응답 y(lose)을 생성할 확률에 대한 odds ratio를 최대화

이 loss의 gradient를 구하면 되는데 밑과 같은 과정을 거침
ln(f(x))의 미분공식과 여러 derivitive 사용 

![image](https://github.com/jinuk0211/rlfh/assets/150532431/be244c65-fb01-4161-8694-f65f566d8245)

![image](https://github.com/jinuk0211/rlfh/assets/150532431/73d6c82e-eac8-415a-aef8-65351237501a)
