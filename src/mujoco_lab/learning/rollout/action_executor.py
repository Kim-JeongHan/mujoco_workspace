# 구현 방향
# policy가 출력한 (B, C, N_a) chunk 중 execution_horizon개를 순서대로 실행한다.
# 남은 행동 큐와 현재 인덱스를 환경별로 관리하고 새 예측 시점을 결정한다.
# 정책 행동의 역정규화 후 env.step에 한 행동씩 전달한다.
# 에피소드 종료 시 남은 chunk를 폐기하고 종료된 환경의 큐만 초기화한다.
# 명령한 행동, 실제 수행한 환경 스텝 수와 종료 결과를 호출 측에 전달한다.
# 초기 PPO에서는 chunk_size=execution_horizon=1로 사용한다.
