"""
Módulo de geração de respostas com LLM.

Integra com OpenAI e Anthropic para gerar respostas
baseadas no contexto recuperado (RAG).
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Resultado da geração de resposta."""

    answer: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    sources: list[dict[str, Any]]

    @property
    def cost_estimate(self) -> float:
        """Estimativa de custo (USD) baseado em preços típicos."""
        # Preços aproximados por 1K tokens
        prices = {
            # OpenAI
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            # Anthropic Claude 4.x
            "claude-opus-4-5-20251101": {"input": 0.015, "output": 0.075},
            "claude-sonnet-4-5-20250929": {"input": 0.003, "output": 0.015},
            # Anthropic Claude 3.x
            "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
            "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
        }

        # Usar preço do claude-3-5-sonnet como default
        price = prices.get(self.model, prices["claude-3-5-sonnet-20241022"])

        input_cost = (self.prompt_tokens / 1000) * price["input"]
        output_cost = (self.completion_tokens / 1000) * price["output"]

        return input_cost + output_cost


class BaseGenerator(ABC):
    """Interface base para geradores de resposta."""

    @abstractmethod
    def generate(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Gera resposta baseada na query e contexto.

        Args:
            query: Pergunta do usuário.
            context: Lista de textos de contexto (chunks recuperados).
            system_prompt: Prompt de sistema customizado.
            **kwargs: Parâmetros adicionais.

        Returns:
            Resultado da geração.
        """
        pass


class OpenAIGenerator(BaseGenerator):
    """
    Gerador usando API da OpenAI.

    Modelos recomendados:
    - gpt-4o: Melhor qualidade, mais caro
    - gpt-4o-mini: Bom custo-benefício (recomendado)
    - gpt-4-turbo: Alta qualidade, contexto longo
    """

    DEFAULT_MODEL = "gpt-4o-mini"

    DEFAULT_SYSTEM_PROMPT = """Você é um assistente especializado em análise de documentos jurídicos e tecnológicos.

INSTRUÇÕES:
1. Responda APENAS com base no contexto fornecido
2. Se a informação não estiver no contexto, diga "Não encontrei essa informação nos documentos"
3. Cite as fontes usando [1], [2], etc.
4. Seja preciso e objetivo
5. Use linguagem técnica apropriada ao domínio

FORMATO DA RESPOSTA:
- Resposta direta e fundamentada
- Citações das fontes relevantes
- Se aplicável, mencione limitações ou ressalvas"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        """
        Inicializa o gerador OpenAI.

        Args:
            api_key: Chave da API (ou usar env var OPENAI_API_KEY).
            model: Nome do modelo.
            temperature: Temperatura (0-1). Menor = mais determinístico.
            max_tokens: Máximo de tokens na resposta.
        """
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = model or self.DEFAULT_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "API key OpenAI não configurada. "
                "Defina OPENAI_API_KEY ou passe api_key."
            )

        self._client = self._init_client()

        logger.info(f"OpenAIGenerator inicializado: model={self._model}")

    def _init_client(self):
        """Inicializa cliente OpenAI."""
        try:
            from openai import OpenAI
            return OpenAI(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Pacote 'openai' não instalado. "
                "Execute: pip install openai"
            )

    def _build_prompt(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None
    ) -> tuple[str, str]:
        """
        Constrói prompts de sistema e usuário.

        Args:
            query: Pergunta.
            context: Contextos recuperados.
            system_prompt: Prompt de sistema customizado.

        Returns:
            Tuple (system_prompt, user_prompt).
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Formatar contexto com numeração
        formatted_context = "\n\n".join([
            f"[{i+1}] {ctx}" for i, ctx in enumerate(context)
        ])

        user = f"""CONTEXTO DOS DOCUMENTOS:
{formatted_context}

PERGUNTA DO USUÁRIO:
{query}

RESPOSTA:"""

        return system, user

    def generate(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
        sources_metadata: list[dict] | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Gera resposta usando OpenAI.

        Args:
            query: Pergunta do usuário.
            context: Lista de textos de contexto.
            system_prompt: Prompt de sistema customizado.
            sources_metadata: Metadados das fontes para citação.
            **kwargs: Parâmetros adicionais (temperature, max_tokens).

        Returns:
            Resultado da geração.
        """
        system, user = self._build_prompt(query, context, system_prompt)

        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.choices[0].message.content
            usage = response.usage

            # Preparar metadados das fontes
            sources = []
            if sources_metadata:
                for i, meta in enumerate(sources_metadata):
                    sources.append({
                        "index": i + 1,
                        "section": meta.get("section", "N/A"),
                        "domain": meta.get("domain", "general"),
                        "score": meta.get("score", 0),
                    })

            return GenerationResult(
                answer=answer,
                model=self._model,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                sources=sources,
            )

        except Exception as e:
            logger.error(f"Erro na geração OpenAI: {e}")
            raise


class AnthropicGenerator(BaseGenerator):
    """
    Gerador usando API da Anthropic (Claude).

    Modelos recomendados:
    - claude-opus-4-5-20251101: Mais capaz (Opus 4.5)
    - claude-sonnet-4-5-20250929: Equilibrado (Sonnet 4.5)
    - claude-3-5-sonnet-20241022: Rápido e econômico
    """

    DEFAULT_MODEL = "claude-opus-4-5-20251101"

    DEFAULT_SYSTEM_PROMPT = """Você é um assistente especializado em análise de documentos jurídicos e tecnológicos.

INSTRUÇÕES:
1. Responda APENAS com base no contexto fornecido
2. Se a informação não estiver no contexto, diga "Não encontrei essa informação nos documentos"
3. Cite as fontes usando [1], [2], etc.
4. Seja preciso e objetivo
5. Use linguagem técnica apropriada ao domínio"""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ) -> None:
        """
        Inicializa o gerador Anthropic.

        Args:
            api_key: Chave da API (ou usar env var ANTHROPIC_API_KEY).
            model: Nome do modelo.
            temperature: Temperatura (0-1).
            max_tokens: Máximo de tokens na resposta.
        """
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._model = model or self.DEFAULT_MODEL
        self._temperature = temperature
        self._max_tokens = max_tokens

        if not self._api_key:
            raise ValueError(
                "API key Anthropic não configurada. "
                "Defina ANTHROPIC_API_KEY ou passe api_key."
            )

        self._client = self._init_client()

        logger.info(f"AnthropicGenerator inicializado: model={self._model}")

    def _init_client(self):
        """Inicializa cliente Anthropic."""
        try:
            import anthropic
            return anthropic.Anthropic(api_key=self._api_key)
        except ImportError:
            raise ImportError(
                "Pacote 'anthropic' não instalado. "
                "Execute: pip install anthropic"
            )

    def generate(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
        sources_metadata: list[dict] | None = None,
        **kwargs
    ) -> GenerationResult:
        """
        Gera resposta usando Anthropic Claude.

        Args:
            query: Pergunta do usuário.
            context: Lista de textos de contexto.
            system_prompt: Prompt de sistema customizado.
            sources_metadata: Metadados das fontes.
            **kwargs: Parâmetros adicionais.

        Returns:
            Resultado da geração.
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Formatar contexto
        formatted_context = "\n\n".join([
            f"[{i+1}] {ctx}" for i, ctx in enumerate(context)
        ])

        user_message = f"""CONTEXTO DOS DOCUMENTOS:
{formatted_context}

PERGUNTA DO USUÁRIO:
{query}

Responda de forma precisa e cite as fontes usando [1], [2], etc."""

        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        try:
            response = self._client.messages.create(
                model=self._model,
                system=system,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            answer = response.content[0].text

            # Preparar metadados das fontes
            sources = []
            if sources_metadata:
                for i, meta in enumerate(sources_metadata):
                    sources.append({
                        "index": i + 1,
                        "section": meta.get("section", "N/A"),
                        "domain": meta.get("domain", "general"),
                        "score": meta.get("score", 0),
                    })

            return GenerationResult(
                answer=answer,
                model=self._model,
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                sources=sources,
            )

        except Exception as e:
            logger.error(f"Erro na geração Anthropic: {e}")
            raise

    def generate_stream(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
        sources_metadata: list[dict] | None = None,
        **kwargs
    ):
        """
        Gera resposta usando streaming.

        Yields:
            Chunks de texto conforme são gerados.
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Formatar contexto
        formatted_context = "\n\n".join([
            f"[{i+1}] {ctx}" for i, ctx in enumerate(context)
        ])

        user_message = f"""CONTEXTO DOS DOCUMENTOS:
{formatted_context}

PERGUNTA DO USUÁRIO:
{query}

Responda de forma precisa e cite as fontes usando [1], [2], etc."""

        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        try:
            with self._client.messages.stream(
                model=self._model,
                system=system,
                messages=[
                    {"role": "user", "content": user_message}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            ) as stream:
                for text in stream.text_stream:
                    yield text

        except Exception as e:
            logger.error(f"Erro no streaming Anthropic: {e}")
            raise

    def generate_stream_with_usage(
        self,
        query: str,
        context: list[str],
        system_prompt: str | None = None,
        sources_metadata: list[dict] | None = None,
        **kwargs
    ) -> tuple[any, dict]:
        """
        Gera resposta com streaming e retorna usage ao final.

        Returns:
            Tuple (generator de chunks, dict para armazenar usage).
        """
        system = system_prompt or self.DEFAULT_SYSTEM_PROMPT

        # Formatar contexto
        formatted_context = "\n\n".join([
            f"[{i+1}] {ctx}" for i, ctx in enumerate(context)
        ])

        user_message = f"""CONTEXTO DOS DOCUMENTOS:
{formatted_context}

PERGUNTA DO USUÁRIO:
{query}

Responda de forma precisa e cite as fontes usando [1], [2], etc."""

        temperature = kwargs.get("temperature", self._temperature)
        max_tokens = kwargs.get("max_tokens", self._max_tokens)

        usage_info = {"input_tokens": 0, "output_tokens": 0}

        def stream_generator():
            try:
                with self._client.messages.stream(
                    model=self._model,
                    system=system,
                    messages=[
                        {"role": "user", "content": user_message}
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                ) as stream:
                    for text in stream.text_stream:
                        yield text

                    # Capturar usage após streaming
                    final_message = stream.get_final_message()
                    usage_info["input_tokens"] = final_message.usage.input_tokens
                    usage_info["output_tokens"] = final_message.usage.output_tokens

            except Exception as e:
                logger.error(f"Erro no streaming Anthropic: {e}")
                raise

        return stream_generator(), usage_info


def create_generator(
    provider: str = "openai",
    **kwargs
) -> BaseGenerator:
    """
    Factory para criar gerador.

    Args:
        provider: Provedor (openai, anthropic).
        **kwargs: Argumentos do gerador.

    Returns:
        Instância do gerador.
    """
    providers = {
        "openai": OpenAIGenerator,
        "anthropic": AnthropicGenerator,
    }

    if provider not in providers:
        raise ValueError(
            f"Provedor '{provider}' não suportado. "
            f"Disponíveis: {list(providers.keys())}"
        )

    return providers[provider](**kwargs)
