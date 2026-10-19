import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import glob
import os
import json

import subprocess
from datetime import datetime
import chromadb
import torch
import doclayout_yolo.nn.tasks
import dashscope
from http import HTTPStatus
torch.serialization.add_safe_globals([doclayout_yolo.nn.tasks.YOLOv10DetectionModel])
from langchain_community.embeddings.dashscope import DashScopeEmbeddings
from mineru.cli.common import do_parse, read_fn
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# 内容提取相关
class MinerURAGSystem:
    def __init__(self, persist_directory="./chroma_db"):
        self.embedding_model = DashScopeEmbeddings(
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-v2"),  # 添加一个默认模型
            dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
        )
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection("documents")

    # 加载文档
    def process_documents(self, pdf_directory, output_dir="./processed"):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        processed_files = []
        for filename in os.listdir(pdf_directory):  # 遍历 PDF 文件夹
            if filename.endswith('.pdf'):
                input_path = os.path.join(pdf_directory, filename)
                output_path = os.path.join(output_dir, filename.replace('.pdf', ''))
                pdf_file_name = Path(input_path).stem
                pdf_bytes = read_fn(input_path)  # 逐个读取 PDF二进制数据
                do_parse(  # 调用 MinerU 的 do_parse 进行智能解析
                    output_dir=output_path,
                    pdf_file_names=[pdf_file_name],
                    pdf_bytes_list=[pdf_bytes],
                    p_lang_list=["ch"],  # 默认使用中文
                    backend="pipeline"
                )
                content_file = os.path.join(output_path, pdf_file_name, "auto", f"{pdf_file_name}_content_list.json")
                # 收集生成的 JSON 文件路径
                if os.path.exists(content_file):
                    processed_files.append(content_file)
        return processed_files  # 返回结果供后续处理

    # 切块
    def _split_into_chunks(self, content_data, chunk_size, overlap):
        chunks = []
        current_chunk = []
        current_length = 0

        for item in content_data:
            if item.get('type') == 'text':
                text = item.get('text', '')
                if not text:
                    continue
                if item.get('text_level', 0) > 0:
                    if current_chunk:
                        chunks.append(
                            {'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text', 'has_title': True}})
                    current_chunk = [f"## {text}"]
                    current_length = len(text)
                elif current_length + len(text) > chunk_size and current_chunk:
                    chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
                    current_chunk = [text]
                    current_length = len(text)
                else:
                    current_chunk.append(text)
                    current_length += len(text)

            elif item.get('type') == 'table':
                if current_chunk:
                    chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
                    current_chunk = []
                    current_length = 0

                table_body = item.get('table_body', '')
                if table_body:
                    if len(table_body) <= chunk_size:
                        chunks.append({
                            'text': table_body,
                            'metadata': {
                                'chunk_type': 'table',
                                'page_idx': item.get('page_idx', 0),
                                'has_caption': len(item.get('table_caption', [])) > 0,
                                'has_footnote': len(item.get('table_footnote', [])) > 0
                            }
                        })
                    else:
                        table_chunks = self._split_table_into_chunks(table_body, chunk_size)
                        for i, chunk in enumerate(table_chunks):
                            chunks.append({
                                'text': chunk,
                                'metadata': {
                                    'chunk_type': 'table',
                                    'page_idx': item.get('page_idx', 0),
                                    'table_chunk_index': i,
                                    'total_table_chunks': len(table_chunks)
                                }
                            })
        if current_chunk:
            chunks.append({'text': ' '.join(current_chunk), 'metadata': {'chunk_type': 'text'}})
        return chunks

    # 表格切块
    def _split_table_into_chunks(self, table_html, chunk_size):
        chunks = []
        if table_html.startswith('<table>'):
            table_content = table_html[7:-8]
        else:
            table_content = table_html
        rows = table_content.split('<tr>')
        table_head = rows[1]
        current_chunk = '<table>'+table_head
        print('表头:', current_chunk)

        current_chunk_size = len(current_chunk)
        for row in rows[2:]:
            row_content = '<tr>' + row
            if current_chunk_size + len(row_content) > chunk_size and current_chunk != '<table>':
                current_chunk += '</table>'
                chunks.append(current_chunk)
                current_chunk = '<table>' +table_head+ row_content
                current_chunk_size = len(current_chunk)
            else:
                current_chunk += row_content
                current_chunk_size += len(row_content)

        if current_chunk != '<table>'+table_head:
            current_chunk += '</table>'
            chunks.append(current_chunk)
        return chunks

    # 向量化
    def chunk_and_embed(self, content_files, chunk_size=512, overlap=50):
        documents = []
        metadatas = []
        ids = []
        for file_path in content_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                content_data = json.load(f)
            doc_metadata = {
                'source': os.path.basename(file_path),
                'file_type': 'pdf',
                'processing_date': datetime.now().isoformat()
            }
            text_chunks = self._split_into_chunks(content_data, chunk_size, overlap)
            for i, chunk in enumerate(text_chunks):
                documents.append(chunk['text'])
                metadatas.append({**doc_metadata, **chunk['metadata']})
                ids.append(f"{os.path.basename(file_path)}_chunk_{i}")
        return documents, metadatas, ids

    # 存储
    def build_vector_store(self, documents, metadatas, ids):
        # 在删除之前，先检查集合中的文档数量是否大于0
        if self.collection.count() > 0:
            print(f"集合不为空，正在删除 {self.collection.count()} 个旧文档...")
            # 获取所有现有文档的ID
            existing_ids = self.collection.get(include=[])['ids']
            # 执行删除操作
            self.collection.delete(ids=existing_ids)
            print("旧文档删除完毕。")
        else:
            print("集合为空，无需删除。")

        # 现在，将新文档添加到干净的集合中
        embeddings = self.embedding_model.embed_documents(documents)
        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"成功存储 {len(documents)} 个文档块到向量数据库")

    # 检索
    def query_documents(self, query_text, n_results=5):
        query_embedding = self.embedding_model.embed_documents([query_text])[0]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        return results


# ==============================================================================
# --- 集成您提供的、功能完善的 RAGQASystem ---
# ==============================================================================
class RAGQASystem:
    def __init__(self, mineru_rag_system):
        self.rag_system = mineru_rag_system
        # 初始化时也为生成模型设置API Key
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

    def _build_prompt(self, question, context_docs):
        context_text = "\n\n".join([f"文档片段 {i + 1}:\n{doc}" for i, doc in enumerate(context_docs)])
        # 优化一下Prompt，让模型更好地理解任务
        prompt = f"""请你扮演一个专业的财务报告分析师。
                    根据下面提供的几段从《厦门灿坤实业股份有限公司2019年年度报告》中摘录的文字，
                    严谨且仅依据这些信息来回答用户的问题。
                    如果提供的信息不足以回答问题，请明确告知“根据现有信息无法回答”。

                    ---
                    相关文档内容：
                    {context_text}
                    ---

                    用户问题：{question}
                    """
        return prompt

    def _call_llm(self, prompt):
        print("\n--- 正在向DashScope LLM发送请求... ---")

        # 使用通义千问模型
        response = dashscope.Generation.call(
            model='qwen-turbo',  # 或者 qwen-plus, qwen-max
            messages=[{'role': 'user', 'content': prompt}],
            result_format='message'  # 设置返回格式为 message
        )

        if response.status_code == HTTPStatus.OK:
            # 请求成功，提取模型的回答
            answer = response.output.choices[0]['message']['content']
            print("--- LLM响应成功 ---")
            return answer
        else:
            # 请求失败，打印错误信息
            print(f"请求失败：request_id={response.request_id}, status_code={response.status_code}, "
                  f"code={response.code}, message={response.message}")
            return "抱歉，调用大语言模型时出错，无法生成答案。"

    def generate_answer(self, question, context):
        prompt = self._build_prompt(question, context)
        # 打印将要发送的最终Prompt，方便调试
        print("\n--- 发送给LLM的最终Prompt ---\n", prompt)
        answer = self._call_llm(prompt)
        return answer, context

    def ask_question(self, question):
        results = self.rag_system.query_documents(question)
        if not results or not results.get('documents') or not results['documents'][0]:
            return "抱歉，没有找到相关的文档信息。", []

        # 将检索到的文档内容传递给 generate_answer
        answer, context = self.generate_answer(question, results['documents'][0])
        return answer, results


# ==============================================================================
# --- 主调用流程 ---
# ==============================================================================
if __name__ == '__main__':

    # 1. 初始化RAG系统
    rag_system = MinerURAGSystem(persist_directory="./my_chroma_db")
    # 2. 解析PDF文档
    # 指定存放PDF的目录
    pdf_folder = 'my_pdfs'
    folder_path = 'processed/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告/2020-03-17__厦门灿坤实业股份有限公司__200512__闽灿坤__2019年__年度报告/auto'
    processed_files = []
    if not os.path.exists(pdf_folder):
        os.makedirs(pdf_folder)
        print(f"请在当前目录下创建'{pdf_folder}'文件夹并放入PDF文件。")
    else:
        # 判断是否已经提取了pdf内容，最耗时的就是解析pdf
        flag = True
        for file in os.listdir(folder_path):
            if file.endswith('.json'):
                print(f"在当前目录下存在json文件。")
                # 使用find命令查找所有_content_list.json文件
                pattern = os.path.join(folder_path, '**', '*_content_list.json')
                processed_files = glob.glob(pattern, recursive=True)
                # 过滤掉空字符串
                processed_files = [f for f in processed_files if f]
                print(f"\n读取了{len(processed_files)}个文件")
                flag = False

        if flag:
            # 解析pdf
            processed_files = rag_system.process_documents(pdf_directory=pdf_folder)
            print(f"解析完成，生成了 {len(processed_files)} 个内容文件。")

    # 2. 直接指定要处理的JSON文件路径
    # 3. 文本分块与向量化准备
    print(f"\n开始{len(processed_files)}个文件进行文本分块与向量化...")
    documents, metadatas, ids = rag_system.chunk_and_embed(content_files=processed_files)
    print(f"已将文档分割成 {len(documents)} 个块。")


    # 4. 构建并填充向量数据库
    if documents:
        print("\n开始构建并填充向量数据库...")
        rag_system.build_vector_store(documents, metadatas, ids)

        # 5. 初始化问答系统
        qa_system = RAGQASystem(mineru_rag_system=rag_system)
        print("\n问答系统已准备就绪。")

        # 6. 提出问题
        # question = ""
        question = "2019年一季度归属于上市公司股东的净利润？"

        print(f"\n--- 正在查询问题 --- \n{question}")
        answer, search_results = qa_system.ask_question(question)

        print("\n\n==================== 问答结果 ====================")
        print("--- 最终回答 ---")
        print(answer)
        print("\n--- 检索到的相关信息 ---")
        if search_results and search_results.get('documents') and search_results['documents'][0]:
            for i, doc in enumerate(search_results['documents'][0]):
                distance = search_results['distances'][0][i]
                metadata = search_results['metadatas'][0][i]
                print(f"\n--- 相关片段 {i + 1} (距离: {distance:.4f}) ---")
                print(f"来源: {metadata.get('source')}")
                print("内容:")
                print(doc)
        else:
            print("没有找到相关的文档片段。")
        print("\n==================================================")
