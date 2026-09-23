# 구현 방향
# Gaussian actor, 별도 critic, vector 환경, rollout buffer와 PPO 알고리즘을 조립한다.
# init_from이면 BC actor와 정규화를 불러오고 critic·optimizer·탐색 설정을 새로 구성한다.
# resume이면 동일 PPO 실험의 학습 상태를 복원한다.
# collector로 현재 정책의 경험을 수집하고 algorithms/ppo.py의 업데이트를 반복 호출한다.
# 초기 구현은 한 번에 한 행동을 선택·실행하고 이후 chunk PPO는 별도 규약으로 확장한다.
# 정규화는 우선 BC 통계로 고정하고 수집 때와 업데이트 때 입력 변환이 같게 한다.
# 평가·로깅·저장 주기와 총 온라인 환경 스텝 예산을 관리한다.
