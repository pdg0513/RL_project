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
```

---

## 2. Task & Data

### 2.1 목표 태스크

사람 손가락(3rd finger, 중지)의 **딱밤 모션 궤적**을 따라가도록  
MyoSuite의 `motorFingerPoseFixed-v0` 환경에서 **continuous control RL (PPO)** 정책을 학습합니다.

### 2.2 타겟 데이터 (Expert Trajectory)

- 파일: `myohand_joint_angles_23dof_stable_signed.npy`
- 형태: `(T_full, 23)` 형태의 23-DoF joint angle 시퀀스
- 이 중 **중지 MCP/PIP/DIP** 에 해당하는 인덱스 `[11, 13, 14]`만 사용 → `(T_full, 3)`
- 프레임 **800 ~ 950** 구간만 잘라서 사용 (딱밤 동작이 들어있는 구간)

---

## 3. Environment & MDP Design

### 3.1 Observation (State)

각 시점의 상태 \(s_t\) 는 7차원 벡터:

- 현재 관절각 `q_t ∈ R^3`  
  - MyoSuite 모델의 `IFmcp`, `IFpip`, `IFdip` (중지 MCP/PIP/DIP)
- 타겟 관절각 `q*_t ∈ R^3`  
  - 위에서 추출한 사람 데이터의 같은 timestep 값
- 정규화된 시간 \(t/T\) (스칼라 1차원)



### 3.2 Action

- MyoSuite `motorFingerPoseFixed-v0` 의 `action_space` 를 그대로 사용
- 연속값 벡터 (각 관절에 대한 torque / control input)

### 3.3 Episode

- 사용 구간: frame **800 ~ 950**
- 코드에서 자동으로

```python
cfg.WINDOW_START = 800
cfg.WINDOW_END   = 950
cfg.EPISODE_LEN  = WINDOW_END - WINDOW_START
```

로 세팅되며, 에피소드 한 번이 이 window를 한 번 쭉 따라가는 구조입니다.

---

## 5. Reward Design

리워드는 크게 세 부분으로 구성됩니다:

\[
r_t = r_{\text{track}} + r_{\text{shape}} + r_{\text{final}}
\]

### 5.1 Tracking Term

타겟 관절각 궤적을 잘 따라가도록 하는 기본 항:

```python
err = q - target
r_track = - cfg.REWARD_SCALE * float(np.sum(err ** 2))
```

- `cfg.REWARD_SCALE` : 기본 L2 tracking 에러의 스케일 (기본 0.05)

### 5.2 Shape Term (딱밤 패턴 보상)

딱밤 모션의 특징인 **초반 flex / 후반 extend** 패턴을 넣기 위해:

1. 타겟 window (`target_angles`) 를 앞/뒤 절반으로 나눔
2. 앞 절반 평균 → `flex_ref` (손가락을 말고 있는 포즈)
3. 뒤 절반 평균 → `ext_ref`  (손가락을 펴고 있는 포즈)

시간 비율 `phi = t / EPISODE_LEN` 에 따라

- `phi < 0.5` → `ref = flex_ref`
- `phi ≥ 0.5` → `ref = ext_ref`

로 두고,

```python
phi = self.t / float(self.cfg.EPISODE_LEN)
ref = self.flex_ref if phi < 0.5 else self.ext_ref

shape_err = q - ref
r_shape = - self.cfg.W_SHAPE * float(np.sum(shape_err ** 2))
```

- `cfg.W_SHAPE` : 패턴(형상) 보상 스케일 (기본 0.10)

### 5.3 Final-state Bonus

마지막 프레임에서 확실하게 펴진 포즈(`ext_ref`)로 끝나도록 하는 보너스:

```python
if self.t == (self.cfg.EPISODE_LEN - 1):
    final_err = q - self.ext_ref
    r_final = - self.cfg.W_FINAL * float(np.sum(final_err ** 2))
else:
    r_final = 0.0
```

- `cfg.W_FINAL` : final-state bonus 스케일 (기본 0.50)

---

## 6. PPO + BC-like Warm Start

### 6.1 PPO 설정

- 알고리즘: **PPO (Proximal Policy Optimization)**
- 기본 하이퍼파라미터:
  - `TOTAL_UPDATES = 10000`
  - `STEPS_PER_UPDATE = 1024`
  - `GAMMA = 0.99`
  - `LAMBDA = 0.95` (GAE)
  - `CLIP_EPS = 0.2`
  - `LR = 1e-5`
  - `BATCH_SIZE = 256`
  - `PPO_EPOCHS = 5`

### 6.2 Actor-Critic Network

- **Actor**
  - 2-layer MLP, hidden dim=128, `Tanh` activation
  - 출력: mean action `μ(s)`
  - `log_std` 를 learnable parameter로 두어 Gaussian 정책 사용
- **Critic**
  - 동일한 구조의 2-layer MLP
  - scalar value \(V(s)\) 출력

### 6.3 BC-like Warm Start

초기 수렴을 돕기 위해 **단순 P-controller** 를 expert처럼 사용해서  
초기 몇 업데이트 동안 actor가 그 방향으로 가까워지도록 보조 loss 추가.

- 휴리스틱 expert action:

```python
q_t   = b_states[:, 0:3]
tgt_t = b_states[:, 3:6]
err_t = tgt_t - q_t

expert_act = torch.zeros_like(ac.actor(b_states))
expert_act[:, 0:3] = cfg.BC_KP * err_t  # 앞 3개 차원에만 사용
```

- BC loss:

```python
mu_pred = ac.actor(b_states)
bc_loss = ((mu_pred - expert_act) ** 2).mean() * cfg.BC_LAMBDA
```

- 최종 PPO loss:

```python
loss = actor_loss + 0.5 * critic_loss + 0.001 * entropy_loss + bc_loss
```

- `update <= cfg.BC_WARM_UPDATES` 인 동안에만 `bc_loss` 활성화

---

## 7. How To Run

### 7.1 단일 실험 (baseline)

```bash
cd src

# NPY 경로가 코드 안에 맞게 설정되어 있는지 확인 후
python motorfinger_ppo_bc.py
```

- 실행하면:
  - `data/myohand_joint_angles_23dof_stable_signed.npy` 를 읽어서 타겟 window 생성
  - `MotorFingerTrajEnv` 래퍼 환경 생성
  - PPO + BC-warm 학습 진행
  - 학습 종료 후:
    - 정책 저장: `ppo_motorfinger_actor_window800_950_bcWarm.pth`
    - 콘솔에 random policy vs trained policy 평가 결과 출력  
      (평균 return, 평균 tracking error)

### 7.2 결과 파일 위치

- 학습된 policy: 프로젝트 루트 또는 `src/` (코드 설정에 따름)
- 결과 figure: `results/figures/` (예시 기준)
  - `ppo_learning_curves_return.png`
  - `ppo_learning_curves_trackerr.png`
  - `ppo_final_tracking_error_bar.png`

---

## 8. Experimental Results

학습 로그 및 평가 결과는 `results/figures` 아래에 시각화된 그래프 형태로 저장됩니다.

### 8.1 Learning Curves – Episode Return

```text
results/figures/ppo_learning_curves_return.png
```

### 8.2 Learning Curves – Tracking Error

```text
results/figures/ppo_learning_curves_trackerr.png
```

### 8.3 Final Policy vs Random Policy

```text
results/figures/ppo_final_tracking_error_bar.png
```

---

## 9. Discussion & Future Work

- 현재 설계된 reward는
  - **궤적 tracking** + **딱밤 패턴(초반 flex / 후반 extend)** + **마지막 frame bonus** 로 구성
  - 하나의 사람 궤적(window 800–950)에 overfit되어 있는 구조
- 향후 개선 아이디어:
  - 여러 딱밤 시도 데이터(여러 trial, 여러 사람)를 넣어서 generalization 강화
  - reward에 관절 속도/가속도 penalty, torque regularization 등을 추가해  
    **더 자연스러운 모션** 유도
  - 다른 RL 알고리즘(SAC, TD3 등)과의 비교 실험
  - 시뮬레이션 결과를 실제 MyoFinger 하드웨어에 transfer 하는 방향 (sim2real)

---

## 10. License

- 연구/실험용 개인 프로젝트 수준이며,  
  MyoSuite 라이선스는 해당 공식 레포를 따릅니다.
- 이 코드 자체는 원하는 라이선스를 추가해서 사용할 수 있습니다.

