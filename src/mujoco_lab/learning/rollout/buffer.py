# 구현 방향
# PPO 한 번의 rollout을 담는 메모리를 (K, E, ...) 형태로 준비한다.
# K는 수집 길이, E는 환경 수이며 state·선택 행동·reward·old value·old log_prob를 저장한다.
# terminated와 truncated, 마지막 관측에서 계산한 bootstrap value를 구분해 보관한다.
# GAE와 return을 담고 업데이트용 미니배치를 생성하며 업데이트 후 buffer를 비운다.
# 확률 계산에 사용한 원래 행동을 보존하고 환경 clipping으로 이를 덮어쓰지 않는다.
# 시연 NPZ 데이터셋과 구분해 현재 정책으로 수집한 경험만 PPO 업데이트에 제공한다.
