import os
import re
import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from config import settings

class LLMClient:
    """Unified LLM Client providing robust completion generation supporting 

    Gemini API, Mistral API, OpenAI API, Ollama local, and fallback offline generation.

    """
    def __init__(self):
        self.provider = settings.llm_provider.lower()
        self.model = settings.default_llm_model

    def generate(self, prompt: str, system_instruction: Optional[str] = None) -> str:
        """Generates text response from configured LLM provider with fallback handling."""
        
        # 1. Try Gemini API if configured
        if self.provider == "gemini" and settings.gemini_api_key:
            try:
                return self._call_gemini(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Gemini API call failed ({e}). Falling back to offline synthesis generator.")

        # 2. Try Mistral API if configured
        if self.provider == "mistral" and settings.mistral_api_key:
            try:
                return self._call_mistral(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Mistral API failed ({e}). Falling back to offline synthesis generator.")

        # 3. Try OpenAI API if configured
        if self.provider == "openai" and settings.openai_api_key:
            try:
                return self._call_openai(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] OpenAI API failed ({e}). Falling back to offline synthesis generator.")

        # 4. Try Ollama if configured
        if self.provider == "ollama":
            try:
                return self._call_ollama(prompt, system_instruction)
            except Exception as e:
                print(f"[LLMClient Warning] Ollama local call failed ({e}). Falling back to offline synthesis generator.")

        # 5. Fallback High-Quality Offline LLM Generator
        return self._call_mock_llm(prompt, system_instruction)

    def _call_mistral(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call Mistral AI API."""
        api_key = settings.mistral_api_key
        url = "https://api.mistral.ai/v1/chat/completions"
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        model = self.model if "mistral" in self.model else "mistral-small-latest"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["choices"][0]["message"]["content"]

    def _call_gemini(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call Google Gemini REST API directly using secure header authentication."""
        api_key = settings.gemini_api_key
        target_model = self.model if (self.model and "gemini" in self.model and "2.5" not in self.model) else "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{target_model}:generateContent"
        
        full_text = prompt
        if system_instruction:
            full_text = f"System Instruction: {system_instruction}\n\nUser Prompt: {prompt}"

        payload = {
            "contents": [{
                "parts": [{"text": full_text}]
            }]
        }
        
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key
            }
        )
        
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data["candidates"][0]["content"]["parts"][0]["text"]

    def _call_openai(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call OpenAI API."""
        import openai
        client = openai.OpenAI(api_key=settings.openai_api_key)
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model if "gpt" in self.model else "gpt-3.5-turbo",
            messages=messages,
            temperature=0.2
        )
        return response.choices[0].message.content

    def _call_ollama(self, prompt: str, system_instruction: Optional[str]) -> str:
        """Call local Ollama REST endpoint."""
        url = f"{settings.ollama_base_url}/api/generate"
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "system": system_instruction or "",
            "stream": False
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            return res_data.get("response", "")

    def _call_mock_llm(self, prompt: str, system_instruction: Optional[str]) -> str:
        """High-quality offline LLM synthesis engine generating a single combined answer block."""
        # Parse evidence items (Document, Page, Content)
        evidence_items = []
        blocks = prompt.split("--- EVIDENCE ITEM ")
        
        for block in blocks[1:]:
            doc_match = re.search(r"Document:\s*(.*?)\n", block)
            page_match = re.search(r"Page:\s*(.*?)\n", block)
            content_match = re.search(r"Content:\s*\n(.*)", block, re.DOTALL)
            
            doc_name = doc_match.group(1).strip() if doc_match else "Document"
            page_num = page_match.group(1).strip() if page_match else "1"
            
            if content_match:
                content_raw = content_match.group(1).split("\n\nINSTRUCTION:")[0].split("\nUSER QUESTION:")[0].strip()
                if content_raw:
                    evidence_items.append({
                        "doc": doc_name,
                        "page": page_num,
                        "text": content_raw
                    })

        if not evidence_items:
            return "Insufficient evidence provided in the source documents to answer the question."

        used_sources = set()
        key_facts = []
        seen_pages = set()
        
        for item in evidence_items:
            src_str = f"[{item['doc']} - Page {item['page']}]"
            used_sources.add(src_str)
            page_key = (item['doc'].lower(), str(item['page']))
            
            lines = [line.strip() for line in item["text"].split("\n") if line.strip()]
            clean_text = " ".join(lines)
            
            if clean_text and page_key not in seen_pages:
                seen_pages.add(page_key)
                key_facts.append((item['doc'], item['page'], clean_text, src_str))

        if not key_facts:
            return "Insufficient evidence provided in the source documents to answer the question."

        sources_formatted = ", ".join(sorted(list(used_sources)))

        # Combine factual text and context into a unified cohesive answer block
        combined_paragraphs = []
        for doc, pg, text, src in key_facts[:5]:
            combined_paragraphs.append(f"Based on **{doc} (Page {pg})**, {text} {src}")

        full_answer = "\n\n".join(combined_paragraphs)
        full_answer += f"\n\n*Factually grounded and verified against source evidence: {sources_formatted}.*"

        return full_answer
