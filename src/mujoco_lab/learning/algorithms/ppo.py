# 구현 방향
# rollout의 reward, value, 종료 정보로 GAE와 return을 계산한다.
# 실제 terminal에서는 bootstrap을 끄고 시간 제한에서는 마지막 관측의 value로 bootstrap한다.
# 두 종료 모두 GAE가 reset 뒤 새 에피소드로 이어지지 않도록 마스크를 처리한다.
# 미니배치에서 저장된 행동의 새 log_prob를 구해 old log_prob와의 확률비를 계산한다.
# clipped policy loss, value loss, entropy 항을 조합하고 여러 epoch 동안 업데이트한다.
# old log_prob·old value·advantage는 gradient 없이 고정하고 배치 shape을 일치시킨다.
# KL, clip fraction, entropy, value loss 등을 반환해 trainer가 학습 상태를 기록하게 한다.
