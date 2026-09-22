"""현재 애플리케이션 진입점.

마이그레이션 중에는 의도적으로 검증된 단일 객체 orchestration에 위임한다.
실제 두 번째 CAD와 회귀 절차를 갖추기 전까지 다중 객체 실행을 보장하지 않는다.
"""

from scripts.test_single_object import main


if __name__ == "__main__":
    raise SystemExit(main())
