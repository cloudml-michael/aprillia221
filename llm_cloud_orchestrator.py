"""
LLM Cloud Orchestrator — manages deployment and routing of multiple
open-source LLMs on AWS, optimizing for cost vs. latency vs. quality.
Built to find the lowest-cost alternative to Bedrock.
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional
import boto3
import json


@dataclass
class LLMEndpoint:
    name: str
    model_id: str
    endpoint_url: str
    cost_per_1k_tokens: float
    avg_latency_ms: float
    max_tokens: int
    instance_type: str
    region: str = "us-east-1"
    is_healthy: bool = True


@dataclass
class InferenceRequest:
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.7
    strategy: str = "cheapest"  # cheapest | fastest | best_quality


@dataclass
class InferenceResult:
    endpoint_name: str
    model_id: str
    response: str
    tokens_used: int
    latency_ms: float
    cost_usd: float


class LLMCloudOrchestrator:
    """
    Routes LLM inference requests across multiple AWS-hosted open-source
    models based on cost, latency, or quality strategy.
    Supports SageMaker endpoints and self-hosted vLLM on EC2.
    """

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self._endpoints: list[LLMEndpoint] = []
        self._sagemaker = boto3.client("sagemaker-runtime", region_name=region)

    def register_endpoint(self, endpoint: LLMEndpoint) -> None:
        self._endpoints.append(endpoint)

    def _select_endpoint(self, strategy: str) -> LLMEndpoint:
        healthy = [e for e in self._endpoints if e.is_healthy]
        if not healthy:
            raise RuntimeError("No healthy endpoints available")
        if strategy == "cheapest":
            return min(healthy, key=lambda e: e.cost_per_1k_tokens)
        elif strategy == "fastest":
            return min(healthy, key=lambda e: e.avg_latency_ms)
        else:  # best_quality — prefer larger models (proxy: max_tokens capacity)
            return max(healthy, key=lambda e: e.max_tokens)

    def invoke(self, request: InferenceRequest) -> InferenceResult:
        endpoint = self._select_endpoint(request.strategy)
        payload = json.dumps({
            "inputs": request.prompt,
            "parameters": {
                "max_new_tokens": request.max_tokens,
                "temperature": request.temperature,
                "do_sample": request.temperature > 0,
            },
        })
        start = time.perf_counter()
        try:
            response = self._sagemaker.invoke_endpoint(
                EndpointName=endpoint.name,
                ContentType="application/json",
                Body=payload,
            )
            result_body = json.loads(response["Body"].read().decode())
            latency_ms = (time.perf_counter() - start) * 1000
            generated = result_body[0].get("generated_text", "") if isinstance(result_body, list) else str(result_body)
            tokens_used = len(generated.split())
            cost = (tokens_used / 1000) * endpoint.cost_per_1k_tokens
            endpoint.avg_latency_ms = 0.9 * endpoint.avg_latency_ms + 0.1 * latency_ms
            return InferenceResult(
                endpoint_name=endpoint.name,
                model_id=endpoint.model_id,
                response=generated,
                tokens_used=tokens_used,
                latency_ms=round(latency_ms, 2),
                cost_usd=round(cost, 6),
            )
        except Exception as exc:
            endpoint.is_healthy = False
            raise RuntimeError(f"Endpoint {endpoint.name} failed: {exc}") from exc

    def health_check(self) -> dict:
        return {
            "total_endpoints": len(self._endpoints),
            "healthy": sum(1 for e in self._endpoints if e.is_healthy),
            "endpoints": [
                {"name": e.name, "model": e.model_id, "healthy": e.is_healthy,
                 "cost_per_1k": e.cost_per_1k_tokens, "avg_latency_ms": round(e.avg_latency_ms, 1)}
                for e in self._endpoints
            ],
        }

    def cost_comparison(self, prompt_tokens: int = 1000) -> list[dict]:
        return sorted([
            {"endpoint": e.name, "model": e.model_id,
             "estimated_cost_usd": round((prompt_tokens / 1000) * e.cost_per_1k_tokens, 6),
             "avg_latency_ms": e.avg_latency_ms}
            for e in self._endpoints
        ], key=lambda x: x["estimated_cost_usd"])
