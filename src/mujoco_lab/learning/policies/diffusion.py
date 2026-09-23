# 구현 방향
# BasePolicy를 상속하고 state로 조건화한 action chunk 생성 모델을 만든다.
# compute_loss는 시연 행동에 시간별 잡음을 추가하고 예측 잡음과 실제 잡음의 MSE를 계산한다.
# 잡음 스케줄과 시간 embedding을 두고 입력 state=(B, N_s), 행동=(B, C, N_a)를 유지한다.
# sample_actions는 가우시안 잡음에서 시작해 학습 스케줄과 맞는 역확산을 수행한다.
# 학습 timestep 수와 추론 step 수를 구분하고 scheduler 설정도 체크포인트에 저장한다.
# 완성 후 factory에 등록하고 같은 ChunkDataset 및 OfflineTrainer로 학습한다.
# 추후 DPPO에는 denoising 경로와 단계별 log_prob를 수집하는 별도 확장이 필요하다.
