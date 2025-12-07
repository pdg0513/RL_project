# RL_project

# MyoFinger PPO Imitation (딱밤 모션 RL 학습)

이 레포는 **MyoSuite** 의 `motorFingerPoseFixed-v0` 환경 위에서  
사람의 중지(3rd finger) **딱밤 모션** 궤적을 따라가도록 PPO로 학습하는 코드입니다.

- 타겟 궤적: `myohand_joint_angles_23dof_stable_signed.npy` 에서  
  중지 MCP/PIP/DIP (index 11, 13, 14)만 뽑은 뒤, **프레임 800–950** 구간.
- 에이전트: motorFinger 관절에 모터 토크를 주는 continuous control 정책 (PPO Actor-Critic).
- 목표: 실제 사람 데이터 기반 궤적을 imitation 하면서,  
  초반에는 손가락을 말고(flex), 후반에는 펴는(extend) 딱밤 패턴을 재현.

---

## 1. Environment & Install

```bash
# (옵션) conda 사용 시
conda create -n myofinger-ppo python=3.9
conda activate myofinger-ppo

pip install -r requirements.txt

# MyoSuite 설치는 공식 레포 참고
# (예) pip install myosuite  또는 git clone 후 pip install -e .

. Problem Setup

환경 (motorFingerPoseFixed-v0)

관절: IFmcp, IFpip, IFdip (중지 MCP, PIP, DIP)

관찰 상태 s_t (dim=7):

현재 관절각 q_t (3)

타겟 관절각 q*_t (3)

정규화된 시간 t / T (1)

액션

motorFinger의 원래 action_space 그대로 사용
(각 관절/모터에 대한 continuous torque command).

에피소드 길이

WINDOW_START=800 ~ WINDOW_END=950 → T ≈ 150

코드는 cfg.EPISODE_LEN = T 로 자동 설정.


4. Reward 설계

에이전트는 매 timestep마다 아래 리워드를 받습니다:

𝑟
𝑡
=
𝑟
track
+
𝑟
shape
+
𝑟
final
r
t
	​

=r
track
	​

+r
shape
	​

+r
final
	​


Tracking term

𝑟
track
=
−
𝜆
track
 
∥
𝑞
𝑡
−
𝑞
𝑡
∗
∥
2
r
track
	​

=−λ
track
	​

∥q
t
	​

−q
t
∗
	​

∥
2

REWARD_SCALE = λ_track

Shape term (딱밤 패턴 보상)
에피소드의 전반부(0~50%)는 말린(flex) 포즈,
후반부(50~100%)는 펴진(extend) 포즈에 가깝도록 유도:

flex_ref = 초반 구간 타겟 평균 (3D)

ext_ref = 후반 구간 타겟 평균 (3D)

𝑟
shape
=
−
𝜆
shape
 
∥
𝑞
𝑡
−
ref
(
𝑡
)
∥
2
r
shape
	​

=−λ
shape
	​

∥q
t
	​

−ref(t)∥
2

W_SHAPE = λ_shape

Final-state bonus
마지막 timestep에서만 fully-extended 포즈에 대한 추가 보상:

r_{\text{final}} = - \lambda_{\text{final}} \, \| q_T - \text{ext_ref} \|^2

W_FINAL = λ_final

5. PPO + BC-like Warm Start

알고리즘: PPO (Proximal Policy Optimization)

gamma=0.99, lambda=0.95, clip_eps=0.2

TOTAL_UPDATES = 10000

STEPS_PER_UPDATE = 1024

batch_size=256, ppo_epochs=5

Actor-Critic: 2-layer MLP (hidden=128, Tanh)

BC-like Warm Start

state 안에 q와 target이 들어 있으므로,

𝑎
expert
≈
𝐾
𝑝
(
𝑞
∗
−
𝑞
)
a
expert
≈K
p
	​

(q
∗
−q)
를 heuristic expert action으로 사용.

업데이트 초반 BC_WARM_UPDATES 동안

actor output μ(s)와 a_expert 사이의 MSE를
PPO 손실에 λ_BC 배로 더해줘서

사람이 가르쳐준 P-controller 근처에서 시작하도록 유도



