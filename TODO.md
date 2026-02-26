# TODO: Идеи для доработки курса

## Выполнено

- [x] Перенести код на использование Mistral API
- [x] Единый pyproject.toml + uv вместо pip/poetry
- [x] Обновить все лабы под LangChain 1.x
- [x] Добавить `.env.example` в git (без реальных ключей)
- [x] Подготовить инструкцию по Langfuse (`langfuse/README.md`)
- [x] Добавить README во все лабораторные работы
- [x] Обновить корневой README с описанием курса и инструкциями

## В планах

- [ ] Интегрировать трассировку Langfuse в лабы:
  - [ ] Добавить CallbackHandler в lab5 (агенты)
  - [ ] Добавить CallbackHandler в lab6 (LangGraph)
  - [ ] Добавить CallbackHandler в lab7 (поисковый агент)
- [ ] Изучить использование Storage напрямую в LangGraph
- [ ] Добавить LLM-интеграцию в lab8 (Telegram-бот):
  - [ ] Подключить поисковый агент из lab7
  - [ ] Добавить веб-поиск через Tavily
