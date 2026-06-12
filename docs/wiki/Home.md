# test_agent Wiki

## 목적
이 문서는 실제 사용자가 챗봇을 바로 실행할 수 있도록 단일 진입점, 인자 전달 방식, 실행 예시를 정리합니다.

## 단일 실행 진입점
- 엔트리 파일: [graph.py](../../graph.py)
- 근거: [graph.py](../../graph.py#L74) 의 __main__ 블록에서 CLI 인자를 받아 실행

## 실행 명령
- python graph.py "질문"

예시:
- python graph.py "삶이란 무엇일까?"
- python graph.py "search: 최신 LangGraph 특징 알려줘"

## 인자 전달 방법
- 질문 인자는 필수입니다.
- 인자가 없으면 Usage 메시지와 함께 종료됩니다.
- 공백이 있는 질문은 큰따옴표로 전달합니다.

## 동작 흐름
1. [graph.py](../../graph.py) 가 질문 문자열을 InputState로 변환
2. [agent_graph/features/rag/graph.py](../../agent_graph/features/rag/graph.py) 에서 RAG 그래프 실행
3. [chatbot.py](../../chatbot.py) 가 LLM을 통해 웹검색(tavily_search) 또는 문서검색(pdf_search) 도구 자동 선택
4. 검색 결과 정리 → 관련성 평가 → 질문 재작성(필요 시) → 답변 생성 → 환각 체크 → 최종 answer 출력
5. 상세 흐름: [RAG 통합 문서](RAG-Integration.md)

## 환경 변수
- OpenAI 키: `OPENAI_API_KEY`
- Tavily 키: `TAVILY_API_KEY`
- Chroma DB 경로: `CHROMA_DB_PATH` (예: `./rag_agent/chroma_db`)
- Chroma 컬렉션: `CHROMA_COLLECTION_NAME` (예: `korean_pdf`)
- 예시 파일: [.env.example](../../.env.example)

## 루트 파일 설명
- [graph.py](../../graph.py): 실행 진입점
- [chatbot.py](../../chatbot.py): 챗봇 노드 (LLM bind_tools 기반)
- [state.py](../../state.py): 상태 스키마
- [llm.py](../../llm.py): 모델 초기화
- [llm_smoke_test.py](../../llm_smoke_test.py): LLM 스모크 테스트

## 관련 문서
- [RAG 통합 설계](RAG-Integration.md)

## 검증
- python -m unittest discover -s tests -v

참고: 가상환경을 사용 중이면 해당 환경을 활성화한 뒤 실행합니다.
