import streamlit as st
import os
import json
import re
import urllib.parse
import requests
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# === 1. НАСТРОЙКИ И СЕКРЕТЫ ===
# Ключ берется из защищенного хранилища Streamlit (secrets)
try:
    YANDEX_API_KEY = st.secrets["YANDEX_API_KEY"]
except:
    YANDEX_API_KEY = ""  # Для локального тестирования, если секреты не настроены

FOLDER_ID = "b1g4b3ft2i3eiql7k3p4"
DB_PATH = "sretensk_db"
TEMPLATES_PATH = "docs/templates"
SITE_INDEX_FILE = "docs/site_index.json"

# === 2. ИНТЕРФЕЙС И СТИЛИ ===
st.set_page_config(page_title="Ассистент СДА", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f4f4f9; }
    .stChatMessage { border-radius: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🎓 Юридический ассистент СДА")
st.caption("Поиск по нормативным актам, положениям и приказам Академии")

# === 3. СИСТЕМНЫЙ ПРОМПТ (ТВОЙ ОРИГИНАЛ) ===
SYSTEM_PROMPT = """
Ты — Интеллектуальный юридический ассистент Сретенской духовной академии (СДА). 
Ты опытный методист с юридическим образованием, работающий в духовном учебном заведении.

КОНТЕКСТ:
- Ты помогаешь студентам, аспирантам и сотрудникам академии
- Твоя задача — давать точные юридические консультации на основе нормативных актов
- Ты должен быть точным, ссылаться на конкретные пункты документов
- Если информации недостаточно — честно об этом скажи

ПРАВИЛА ОТВЕТА ( СТРОГО ):

1. СТРУКТУРА ОТВЕТА ЮРИСТА:

📌 **ЗАКЛЮЧЕНИЕ** (1-2 предложения)[Прямой ответ: ДА/НЕТ/ТРЕБУЕТСЯ/В ЗАВИСИМОСТИ ОТ...]

📖 **ПРАВОВОЕ ОБОСНОВАНИЕ**
   [Развернутый анализ со ссылками на конкретные пункты]

   Пример: "Согласно п. 3.2 Положения о порядке предоставления академических отпусков..."
   или "В соответствии со ст. 28 Федерального закона..."

📋 **ПОРЯДОК ДЕЙСТВИЙ** (если применимо)
   1. [Первый шаг]
   2.[Второй шаг]
   3. [Срок/условия]

📎 **ДОКУМЕНТЫ**
   • [Название документа, номер, пункт]

🔗 **ССЫЛКИ НА САЙТ**
   • [Проверенная ссылка на страницу сайта sdamp.ru, если релевантно]

2. ГЛУБОКИЙ ПОИСК:
   - Анализируй ВСЕ найденные фрагменты
   - Если документ ссылается на другой — найди его
   - Объединяй информацию из разных источников

3. ПРОВЕРКА ССЫЛОК (ОБЯЗАТЕЛЬНО!):
   - Если упоминаешь ссылку на сайт sdamp.ru — ОБЯЗАТЕЛЬНО проверь, что она существует
   - НЕ выдумывай ссылки — используй ТОЛЬКО те, что найдены в контексте или в индексе сайта
   - Если ссылка не найдена в документах — НЕ включай её в ответ
   - Лучше сказать " ссылку уточните на сайте" чем дать нерабочую ссылку

4. ВАЖНО:
   - Используй ТОЛЬКО информацию из документов
   - Не давай общих советов без ссылок на нормативку
   - Если нужен образец заявления — предложи шаблон
   - НЕ галлюцинируй — не придумывай ссылки и документы

5. УТОЧНЯЮЩИЕ ВОПРОСЫ (ОБЯЗАТЕЛЬНО!):
   В КОНЦЕ каждого ответа добавь 2-3 уточняющих вопроса, которые могут заинтересовать пользователя.
   Формат вывода строго:

   🎯 УТОЧНЯЮЩИЕ ВОПРОСЫ:
   [Вопрос 1?] [Вопрос 2?][Вопрос 3?]

Стиль: Профессиональный юридический, но доброжелательный.
"""


# === 4. ЗАГРУЗКА БАЗЫ (КЕШИРОВАНИЕ ДЛЯ СКОРОСТИ) ===
@st.cache_resource
def load_resources():
    # Загрузка индекса сайта
    site_index = {'pages': [], 'documents': []}
    if os.path.exists(SITE_INDEX_FILE):
        try:
            with open(SITE_INDEX_FILE, 'r', encoding='utf-8') as f:
                site_index = json.load(f)
        except:
            pass

    # Загрузка векторной базы
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)

    templates_db_path = DB_PATH + "_templates"
    if os.path.exists(templates_db_path):
        db_templates = FAISS.load_local(templates_db_path, embeddings, allow_dangerous_deserialization=True)
        db.merge_from(db_templates)

    return db, site_index


db, site_index = load_resources()


# === 5. ТВОИ ЛОГИЧЕСКИЕ ФУНКЦИИ (ПОЛНОСТЬЮ СОХРАНЕНЫ) ===
def clean_document_name(filename: str) -> str:
    name = urllib.parse.unquote(filename)
    name = re.sub(r'\.(docx?|pdf|txt)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'([а-яёa-z])([А-ЯЁA-Z])', r'\1 \2', name)
    name = re.sub(r'^[\d\s]*СДА\s*', '', name, flags=re.IGNORECASE)
    name = re.sub(r'^[\d\.\-\s]+', '', name)
    name = re.sub(r'(?<!от\s)(\d{2})[-_.](\d{2})[-_.](\d{4})', r'от \1.\2.\3', name)
    name = re.sub(r'\s+', ' ', name).strip()
    return name if name else filename


def extract_keywords(query: str) -> list:
    stop_words = {'как', 'что', 'где', 'когда', 'почему', 'можно', 'нужно', 'могу', 'ли', 'или', 'и', 'в', 'на', 'по',
                  'для', 'при', 'о', 'об'}
    words = re.findall(r'\b[а-яёА-ЯЁ]{4,}\b', query.lower())
    return [w for w in words if w not in stop_words]


def extract_document_references(docs: list) -> list:
    references = []
    patterns = [r'[Пп]оложение[а-яё\s]*["«]([^"]+)["»]', r'[Пп]риказ[а-яё\s]*№?\s*\d+.*["«]([^"]+)["»]']
    for doc in docs:
        for pattern in patterns:
            references.extend(re.findall(pattern, doc['content']))
    return list(set(references))[:10]


def find_link_in_index(query: str) -> list:
    query_lower = query.lower()
    results = []
    for page in site_index.get('pages', []):
        if query_lower in page.get('title', '').lower() or query_lower in page.get('url', '').lower():
            results.append({'title': page.get('title', 'Страница'), 'url': page.get('url', '')})
    for doc in site_index.get('documents', []):
        if query_lower in doc.get('name', '').lower():
            results.append({'title': doc.get('name', 'Документ'), 'url': doc.get('url', '')})
    return results[:5]


def find_template(user_query: str) -> str | None:
    if not os.path.exists(TEMPLATES_PATH):
        return None
    templates = os.listdir(TEMPLATES_PATH)
    query_lower = user_query.lower()
    keywords_map = {
        'академ': ['академ', 'академическ'],
        'отчисл': ['отчисл', 'выбыт', 'уход'],
        'пересдач': ['пересдач', 'оценк'],
        'дистан': ['дистан', 'онлайн'],
        'справк': ['справк', 'архив'],
        'общежити': ['общежити', 'жиль'],
    }
    for _, search_terms in keywords_map.items():
        if any(term in query_lower for term in search_terms):
            for t in templates:
                if any(term in t.lower() for term in search_terms):
                    return os.path.join(TEMPLATES_PATH, t)
    return None


def parse_suggestions(answer: str) -> list:
    suggestions = []
    patterns = [r'🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)', r'УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)',
                r'💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?(.+)']
    for pattern in patterns:
        match = re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        if match:
            suggestions_text = match.group(1).strip()
            questions = re.findall(r'\[([^\]]+)\]|\b([А-Яа-яёЁ].*?\?)', suggestions_text)
            for q in questions:
                if isinstance(q, tuple):
                    for part in q:
                        if part.strip(): suggestions.append(part.strip())
                elif q.strip():
                    suggestions.append(q.strip())
            break
    if not suggestions:
        lines = answer.split('\n')
        for line in reversed(lines[-5:]):
            if '?' in line and len(line) < 100:
                question = re.sub(r'^\d+[\.\)]\s*', '', line.strip())
                if question and question not in suggestions: suggestions.append(question)
    return suggestions[:3]


def clean_answer(answer: str) -> str:
    patterns = [r'\n🎯\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+', r'\nУТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+',
                r'\n💡\s*УТОЧНЯЮЩИЕ\s*ВОПРОСЫ[:\s]*\n?.+']
    for pattern in patterns:
        answer = re.sub(pattern, '', answer, flags=re.DOTALL)
    return answer.strip()


# === 6. ЛОГИКА RAG И ЗАПРОС К YANDEX ===
def iterative_search(query: str):
    found_docs = []
    sources_set = set()

    docs_stage1 = db.similarity_search(query, k=12)
    for d in docs_stage1:
        raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
        source = clean_document_name(raw_source)
        sources_set.add(source)
        found_docs.append({'source': source, 'content': d.page_content, 'stage': 1})

    for term in extract_keywords(query)[:3]:
        docs_stage2 = db.similarity_search(term, k=6)
        for d in docs_stage2:
            raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
            source = clean_document_name(raw_source)
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 2})

    for doc_ref in extract_document_references(found_docs)[:5]:
        docs_stage3 = db.similarity_search(doc_ref, k=4)
        for d in docs_stage3:
            raw_source = os.path.basename(d.metadata.get('source', 'Неизвестный'))
            source = clean_document_name(raw_source)
            if source not in [doc['source'] for doc in found_docs]:
                sources_set.add(source)
                found_docs.append({'source': source, 'content': d.page_content, 'stage': 3})

    return found_docs, sources_set


def get_rag_response(question: str):
    docs, sources = iterative_search(question)
    site_links = find_link_in_index(question)

    site_context = ""
    if site_links:
        site_context = "\n📎 РЕЛЕВАНТНЫЕ ССЫЛКИ НА САЙТЕ (ПРОВЕРЕННЫЕ):\n"
        for link in site_links:
            site_context += f"- {link['title']}: {link['url']}\n"

    if not docs:
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            return f"В базе документов не найдено, но есть информация на сайте:\n{links_text}", [], []
        return "😔 В базе знаний не найдено релевантных документов.", [], []

    docs.sort(key=lambda x: x['stage'])
    context = "\n\n".join(
        [f"--- ФРАГМЕНТ {i + 1} ({d['source']}) ---\n{d['content']}" for i, d in enumerate(docs[:15])])

    if site_context: context += site_context

    # Обращение к YandexGPT
    url = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    headers = {"Authorization": f"Api-Key {YANDEX_API_KEY}", "x-folder-id": FOLDER_ID}
    payload = {
        "modelUri": f"gpt://{FOLDER_ID}/yandexgpt/latest",
        "completionOptions": {"stream": False, "temperature": 0.2, "maxTokens": "2000"},
        "messages": [
            {"role": "system", "text": SYSTEM_PROMPT},
            {"role": "user",
             "text": f"КОНТЕКСТ:\n{context}\n\nВОПРОС: {question}\n\nДай структурированный ответ с уточняющими вопросами в конце. Проверь, что все ссылки реальны!"}
        ]
    }

    try:
        res = requests.post(url, headers=headers, json=payload)
        if res.status_code != 200:
            return f"Ошибка YandexGPT: {res.text}", [], []

        answer = res.json()['result']['alternatives'][0]['message']['text']
        suggestions = parse_suggestions(answer)
        answer = clean_answer(answer)

        sources_text = "\n".join([f"• {s}" for s in sources])
        if site_links:
            links_text = "\n".join([f"🔗 [{link['title']}]({link['url']})" for link in site_links])
            sources_text += f"\n\n🌐 *Проверенные ссылки на сайт:*\n{links_text}"

        return answer, suggestions, sources_text
    except Exception as e:
        return f"Произошла ошибка при обращении к ИИ: {e}", [], []


# === 7. ИНТЕРФЕЙС И ЛОГИКА ЧАТА СТРИМЛИТ ===
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant",
                                  "content": "👋 Здравствуйте! Я знаю всё о Положениях, Приказах и Уставе Академии. Чем могу помочь?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Введите ваш вопрос..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Изучаю документы Академии..."):
            # 1. Проверяем, нужен ли шаблон
            template_path = find_template(prompt)
            if template_path:
                st.markdown(f"📄 **Нашёл подходящий шаблон:** {os.path.basename(template_path)}")
                # Кнопка скачивания шаблона прямо в чате!
                try:
                    with open(template_path, "rb") as file:
                        st.download_button(
                            label="📥 Скачать документ",
                            data=file,
                            file_name=os.path.basename(template_path),
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                        )
                except:
                    st.error("Файл шаблона временно недоступен.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": f"Предложен шаблон: {os.path.basename(template_path)}"})

            else:
                # 2. Обычный вопрос к RAG
                answer, suggestions, sources_text = get_rag_response(prompt)

                # Вывод ответа
                st.markdown(answer)

                if sources_text:
                    with st.expander("📚 Использованные документы"):
                        st.markdown(sources_text)

                if suggestions:
                    st.markdown("🎯 **Уточняющие вопросы:**")
                    for sug in suggestions:
                        st.button(sug, key=sug)  # В Streamlit это создаст интерактивные кнопки

                st.session_state.messages.append({"role": "assistant", "content": answer})