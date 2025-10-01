import streamlit as st
import os
from dotenv import load_dotenv
import assemblyai as aai
import voyageai
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel
import google.genai as genai
import tempfile
import uuid

load_dotenv()


@st.cache_resource
def init_services():
    api_key = os.getenv("ASSEMBLYAI_API_KEY")
    aai.settings.api_key = api_key

    vo = voyageai.Client(api_key=os.getenv("VOYAGE_API_KEY"))

    connection_string = os.getenv("MONGODB_CONNECTION_STRING")
    # 增強 MongoDB 連接設定
    client = MongoClient(connection_string)
    collection = client["rag_db"]["test"]

    llm = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    return vo, collection, llm


def transcribe_audio(audio_file_path):
    config = aai.TranscriptionConfig(
        speech_model=aai.SpeechModel.best,
        language_code='zh',
        speaker_labels=True
    )

    transcript = aai.Transcriber(config=config).transcribe(audio_file_path)

    speaker_transcripts = []
    for utterance in transcript.utterances:
        speaker_transcripts.append({
            "speaker": f"Speaker {utterance.speaker}",
            "text": utterance.text
        })

    return speaker_transcripts


def create_embeddings_and_store(speaker_transcripts, vo, collection):

    speaker_sents = [f"{item['speaker']}: {item['text']}" for item in speaker_transcripts]

    embeds = vo.embed(
        texts=speaker_sents,
        model='voyage-2',
        input_type='document'
    ).embeddings

    docs = []
    session_id = str(uuid.uuid4())

    for (transcript, embed) in zip(speaker_sents, embeds):
        docs.append({
            "text": transcript,
            "embedding": embed,
            "session_id": session_id
        })

    # 插入文檔到 MongoDB
    try:
        result = collection.insert_many(docs)
        st.info(f"已成功插入 {len(docs)} 個文檔")
    except Exception as e:
        st.error(f"插入文檔時發生錯誤: {str(e)}")
        raise

    # 創建向量索引（如果不存在）
    try:
        search_index_model = SearchIndexModel(
            definition={
                "fields": [
                    {
                        "type": "vector",
                        "numDimensions": 1024,
                        "path": "embedding",
                        "similarity": "cosine"
                    }
                ]
            },
            name="vector_index",
            type="vectorSearch"
        )
        
        existing_indexes = list(collection.list_search_indexes())
        vector_index_exists = any(idx.get('name') == 'vector_index' for idx in existing_indexes)
        
        if not vector_index_exists:
            collection.create_search_index(model=search_index_model)
            st.info("已創建向量索引")
        else:
            st.info("向量索引已存在")
            
    except Exception as e:
        st.warning(f"索引操作時發生錯誤: {str(e)}")
        st.info("數據已成功儲存，索引可能需要稍後創建")

    return session_id, len(docs)


def search_and_generate_response(query, vo, collection, llm, session_id=None):
    query_embed = vo.embed(
        [query],
        model='voyage-2',
        input_type='query'
    ).embeddings[0]

    search_pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "queryVector": query_embed,
                "path": "embedding",
                "limit": 5,
                "exact": True
            }
        },
        {
            "$project": {
                "_id": 0,
                "text": 1,
                "session_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]

    if session_id:
        search_pipeline.insert(1, {
            "$match": {"session_id": session_id}
        })

    try:
        results = list(collection.aggregate(search_pipeline))
    except Exception as e:
        error_msg = str(e)
        return f"搜索時發生錯誤: {error_msg}", []

    if not results:
        return "抱歉，我在資料庫中找不到相關資訊。", []

    merged_context = "\n\n---\n\n".join([item['text'] for item in results])

    prompt = (
        f"Context information is below.\n"
        f"----------------------- \n"
        f"{merged_context}\n"
        f"----------------------- \n"
        f"Given the context information above, think step by step "
        f"to answer the query in a crisp manner. Please answer in Traditional Chinese.\n"
        f"Query: {query}\n"
        f"Answer: "
    )

    try:
        response = llm.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        answer = response.candidates[0].content.parts[0].text
    except Exception as e:
        answer = f"生成回答時發生錯誤: {str(e)}"

    return answer, results


def main():
    st.set_page_config(
        page_title="RAG-語音內容檢索",
        page_icon="🎵",
        layout="wide"
    )

    st.title("RAG-語音內容檢索")
    st.markdown("上傳音訊檔案，讓AI幫您分析和回答問題")

    try:
        vo, collection, llm = init_services()
        st.success("所有服務已成功連接")
    except Exception as e:
        st.error(f"服務初始化失敗: {str(e)}")
        st.info("請檢查您的環境變數設定 (ASSEMBLYAI_API_KEY, VOYAGE_API_KEY, MONGODB_CONNECTION_STRING, GEMINI_API_KEY)")
        return

    with st.sidebar:
        st.header("音訊檔案上傳")
        
        # 添加手動清理按鈕
        if st.button("一鍵清除資料庫", help="清空所有資料並重建正確的向量索引"):
            try:
                with st.spinner("正在清理資料庫..."):
                    # 清空資料
                    collection.delete_many({})
                    
                    # 刪除舊索引
                    try:
                        collection.drop_search_index("vector_index")
                        st.info("已刪除舊索引")
                    except Exception as e:
                        st.info("索引不存在或已被刪除")
                    
                    # 等待一段時間讓索引完全刪除
                    import time
                    time.sleep(2)
                    
                    st.success("資料庫已清理完成，請重新上傳音訊檔案")
                    
                    # 清除 session state
                    if 'session_id' in st.session_state:
                        del st.session_state.session_id
                    if 'transcripts' in st.session_state:
                        del st.session_state.transcripts
                    if 'chat_history' in st.session_state:
                        del st.session_state.chat_history
                        
            except Exception as e:
                st.error(f"清理資料庫時發生錯誤: {str(e)}")

        uploaded_file = st.file_uploader(
            "選擇音訊檔案",
            type=['mp3', 'wav', 'm4a', 'flac', 'ogg'],
            help="支援格式: MP3, WAV, M4A, FLAC, OGG"
        )

        if uploaded_file is not None:
            st.audio(uploaded_file, format='audio/wav')
            
            if st.button("開始處理音訊", type="primary"):
                # 創建臨時檔案
                with tempfile.NamedTemporaryFile(delete=False,
                                                 suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                try:
                    # 步驟1: 轉錄音訊
                    with st.spinner("正在轉錄音訊..."):
                        speaker_transcripts = transcribe_audio(tmp_file_path)
                        st.session_state.transcripts = speaker_transcripts
                    
                    st.success("音訊轉錄完成")
                    
                    # 步驟2: 生成向量嵌入並儲存
                    with st.spinner("正在生成嵌入向量並儲存到資料庫..."):
                        session_id, doc_count = create_embeddings_and_store(
                            speaker_transcripts, vo, collection
                        )
                        st.session_state.session_id = session_id
                    
                    st.success(f"已成功處理並儲存 {doc_count} 個文檔片段")

                except Exception as e:
                    st.error(f"處理音訊時發生錯誤: {str(e)}")

                finally:
                    os.unlink(tmp_file_path)

        if 'transcripts' in st.session_state:
            st.header("轉錄結果")
            with st.expander("查看完整轉錄", expanded=False):
                for i, item in enumerate(st.session_state.transcripts):
                    st.markdown(f"**{item['speaker']}:** {item['text']}")

    st.header("AI 助手")

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if query := st.chat_input("請輸入您的問題..."):
        if 'session_id' not in st.session_state:
            st.warning("請先上傳並處理音訊檔案")
            return

        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("正在思考中..."):
                try:
                    answer, search_results = search_and_generate_response(
                        query, vo, collection, llm, st.session_state.session_id
                    )

                    st.markdown(answer)

                    if search_results:
                        with st.expander("📚 相關資訊來源", expanded=False):
                            for i, result in enumerate(search_results):
                                score = result.get('score', 'N/A')
                                st.markdown(f"**來源 {i + 1}** (相似度: {score:.3f}):")
                                st.markdown(f"```\n{result['text']}\n```")

                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                except Exception as e:
                    error_msg = f"生成回答時發生錯誤: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})

    if st.session_state.chat_history:
        if st.button("清除聊天記錄"):
            st.session_state.chat_history = []
            st.rerun()


if __name__ == "__main__":
    main()
