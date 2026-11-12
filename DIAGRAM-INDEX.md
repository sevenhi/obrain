# Diagram Index

Automatic index of every `diagrams/*.html`; receipts add validation status.
Regenerate with `python3 tools/build_diagram_index.py`.
Legacy diagrams without receipts are explicitly marked `legacy-unverified`.

## Architecture (16)

| Diagram | Quality | Validation |
|---|---|---|
| [cl-gmma-descriptor.architecture](diagrams/cl-gmma-descriptor.architecture.html) | showcase | receipt-verified |
| [di-dg-backend-mapping.architecture](diagrams/di-dg-backend-mapping.architecture.html) | showcase | receipt-verified |
| [di-engine-layers.architecture](diagrams/di-engine-layers.architecture.html) | showcase | receipt-verified |
| [di-epd-topology.architecture](diagrams/di-epd-topology.architecture.html) | showcase | receipt-verified |
| [di-v3-3d-partition.architecture](diagrams/di-v3-3d-partition.architecture.html) | showcase | receipt-verified |
| [di-v3-parallel-primitives.architecture](diagrams/di-v3-parallel-primitives.architecture.html) | showcase | receipt-verified |
| [fa-warp-partition.architecture](diagrams/fa-warp-partition.architecture.html) | showcase | receipt-verified |
| [flash-attention-implementation.architecture](diagrams/flash-attention-implementation.architecture.html) | unknown | legacy-unverified |
| [ma-ds-kv-compare.architecture](diagrams/ma-ds-kv-compare.architecture.html) | showcase | receipt-verified |
| [ma-mha-mqa-gqa-heads.architecture](diagrams/ma-mha-mqa-gqa-heads.architecture.html) | showcase | receipt-verified |
| [ma-v3-node-limited.architecture](diagrams/ma-v3-node-limited.architecture.html) | showcase | receipt-verified |
| [rng-thread-vs-block.architecture](diagrams/rng-thread-vs-block.architecture.html) | showcase | receipt-verified |
| [vllm-index.architecture](diagrams/vllm-index.architecture.html) | unknown | legacy-unverified |
| [vm-pd-topology.architecture](diagrams/vm-pd-topology.architecture.html) | showcase | receipt-verified |
| [vm-quant-by-layer.architecture](diagrams/vm-quant-by-layer.architecture.html) | showcase | receipt-verified |
| [vm-v1-three-layers.architecture](diagrams/vm-v1-three-layers.architecture.html) | showcase | receipt-verified |

## Dataflow (40)

| Diagram | Quality | Validation |
|---|---|---|
| [at-aiayn-scaled-dot.dataflow](diagrams/at-aiayn-scaled-dot.dataflow.html) | showcase | receipt-verified |
| [at-laser-logsumexp.dataflow](diagrams/at-laser-logsumexp.dataflow.html) | showcase | receipt-verified |
| [cl-layout-composition.dataflow](diagrams/cl-layout-composition.dataflow.html) | showcase | receipt-verified |
| [cl-layout-to-index.dataflow](diagrams/cl-layout-to-index.dataflow.html) | showcase | receipt-verified |
| [cl-mainloop-epilogue.dataflow](diagrams/cl-mainloop-epilogue.dataflow.html) | showcase | receipt-verified |
| [cl-minimal-gemm-path.dataflow](diagrams/cl-minimal-gemm-path.dataflow.html) | showcase | receipt-verified |
| [cl-tcgen05-tmem.dataflow](diagrams/cl-tcgen05-tmem.dataflow.html) | showcase | receipt-verified |
| [cl-wgmma-fragments.dataflow](diagrams/cl-wgmma-fragments.dataflow.html) | showcase | receipt-verified |
| [cv-deploy-pipeline.dataflow](diagrams/cv-deploy-pipeline.dataflow.html) | showcase | receipt-verified |
| [cv-faster-mask-rcnn.dataflow](diagrams/cv-faster-mask-rcnn.dataflow.html) | showcase | receipt-verified |
| [cv-numeric-layers.dataflow](diagrams/cv-numeric-layers.dataflow.html) | showcase | receipt-verified |
| [cv-vit-patchify.dataflow](diagrams/cv-vit-patchify.dataflow.html) | showcase | receipt-verified |
| [di-dg-jit-flow.dataflow](diagrams/di-dg-jit-flow.dataflow.html) | showcase | receipt-verified |
| [di-kv-tiers.dataflow](diagrams/di-kv-tiers.dataflow.html) | showcase | receipt-verified |
| [di-sinkhorn-projection.dataflow](diagrams/di-sinkhorn-projection.dataflow.html) | showcase | receipt-verified |
| [di-zos-seqlens-fix.dataflow](diagrams/di-zos-seqlens-fix.dataflow.html) | showcase | receipt-verified |
| [dm-esft-adapter-storage.dataflow](diagrams/dm-esft-adapter-storage.dataflow.html) | showcase | receipt-verified |
| [dv-ocr-optical-compression.dataflow](diagrams/dv-ocr-optical-compression.dataflow.html) | showcase | receipt-verified |
| [dv-vl-hybrid-dual-tower.dataflow](diagrams/dv-vl-hybrid-dual-tower.dataflow.html) | showcase | receipt-verified |
| [fa-fp8-precision.dataflow](diagrams/fa-fp8-precision.dataflow.html) | showcase | receipt-verified |
| [fa-standard-attention-hbm.dataflow](diagrams/fa-standard-attention-hbm.dataflow.html) | showcase | receipt-verified |
| [fp8-fp4-formats.dataflow](diagrams/fp8-fp4-formats.dataflow.html) | unknown | legacy-unverified |
| [ma-ds-mla-kv.dataflow](diagrams/ma-ds-mla-kv.dataflow.html) | showcase | receipt-verified |
| [ma-gqa-vs-mla-compression.dataflow](diagrams/ma-gqa-vs-mla-compression.dataflow.html) | showcase | receipt-verified |
| [ma-v2-v3-combo.dataflow](diagrams/ma-v2-v3-combo.dataflow.html) | showcase | receipt-verified |
| [ma-v3-mtp.dataflow](diagrams/ma-v3-mtp.dataflow.html) | showcase | receipt-verified |
| [qt-fakequant-fold-weight.dataflow](diagrams/qt-fakequant-fold-weight.dataflow.html) | showcase | receipt-verified |
| [qt-moe-gptq-memory-formula.dataflow](diagrams/qt-moe-gptq-memory-formula.dataflow.html) | showcase | receipt-verified |
| [qt-quip-sharp-pipeline.dataflow](diagrams/qt-quip-sharp-pipeline.dataflow.html) | showcase | receipt-verified |
| [rng-counter-based.dataflow](diagrams/rng-counter-based.dataflow.html) | showcase | receipt-verified |
| [vm-int4-packing.dataflow](diagrams/vm-int4-packing.dataflow.html) | showcase | receipt-verified |
| [vm-llmc-ct-save.dataflow](diagrams/vm-llmc-ct-save.dataflow.html) | showcase | receipt-verified |
| [vm-omni-kv-transfer.dataflow](diagrams/vm-omni-kv-transfer.dataflow.html) | showcase | receipt-verified |
| [vm-pd-request-path.dataflow](diagrams/vm-pd-request-path.dataflow.html) | showcase | receipt-verified |
| [vm-prodstack-lmcache.dataflow](diagrams/vm-prodstack-lmcache.dataflow.html) | showcase | receipt-verified |
| [vm-spec-vocab-map.dataflow](diagrams/vm-spec-vocab-map.dataflow.html) | showcase | receipt-verified |
| [vm-staged-write.dataflow](diagrams/vm-staged-write.dataflow.html) | showcase | receipt-verified |
| [vm-static-vs-continuous.dataflow](diagrams/vm-static-vs-continuous.dataflow.html) | showcase | receipt-verified |
| [vm-token-bin-count.dataflow](diagrams/vm-token-bin-count.dataflow.html) | showcase | receipt-verified |
| [vm-vime-memory-topology.dataflow](diagrams/vm-vime-memory-topology.dataflow.html) | showcase | receipt-verified |

## Sequence (10)

| Diagram | Quality | Validation |
|---|---|---|
| [cl-collective-builder.sequence](diagrams/cl-collective-builder.sequence.html) | showcase | receipt-verified |
| [di-mhc-cuda-graph-race.sequence](diagrams/di-mhc-cuda-graph-race.sequence.html) | showcase | receipt-verified |
| [fa-nan-pollution.sequence](diagrams/fa-nan-pollution.sequence.html) | showcase | receipt-verified |
| [vm-enginecore-step.sequence](diagrams/vm-enginecore-step.sequence.html) | showcase | receipt-verified |
| [vm-omni-qwen3-async.sequence](diagrams/vm-omni-qwen3-async.sequence.html) | showcase | receipt-verified |
| [vm-router-pd-sequence.sequence](diagrams/vm-router-pd-sequence.sequence.html) | showcase | receipt-verified |
| [vm-sr-decision-flow.sequence](diagrams/vm-sr-decision-flow.sequence.html) | showcase | receipt-verified |
| [vm-v1-async-schedule.sequence](diagrams/vm-v1-async-schedule.sequence.html) | showcase | receipt-verified |
| [vm-v1-request-journey.sequence](diagrams/vm-v1-request-journey.sequence.html) | showcase | receipt-verified |
| [vm-vime-train-loop.sequence](diagrams/vm-vime-train-loop.sequence.html) | showcase | receipt-verified |

## Workflow (168)

| Diagram | Quality | Validation |
|---|---|---|
| [at-aiayn-mask.workflow](diagrams/at-aiayn-mask.workflow.html) | showcase | receipt-verified |
| [at-aiayn-multihead.workflow](diagrams/at-aiayn-multihead.workflow.html) | showcase | receipt-verified |
| [at-backend-selection.workflow](diagrams/at-backend-selection.workflow.html) | showcase | receipt-verified |
| [at-blockmask-three-classes.workflow](diagrams/at-blockmask-three-classes.workflow.html) | showcase | receipt-verified |
| [at-fa4-two-hooks.workflow](diagrams/at-fa4-two-hooks.workflow.html) | showcase | receipt-verified |
| [at-flexattention-pipeline.workflow](diagrams/at-flexattention-pipeline.workflow.html) | showcase | receipt-verified |
| [at-laser-gradient.workflow](diagrams/at-laser-gradient.workflow.html) | showcase | receipt-verified |
| [at-lse-sequence-merge.workflow](diagrams/at-lse-sequence-merge.workflow.html) | showcase | receipt-verified |
| [at-scoremod-maskmod-why-split.workflow](diagrams/at-scoremod-maskmod-why-split.workflow.html) | showcase | receipt-verified |
| [at-sink-baselines.workflow](diagrams/at-sink-baselines.workflow.html) | showcase | receipt-verified |
| [at-sink-cache-layout.workflow](diagrams/at-sink-cache-layout.workflow.html) | showcase | receipt-verified |
| [at-sink-causal-chain.workflow](diagrams/at-sink-causal-chain.workflow.html) | showcase | receipt-verified |
| [at-spatten-cascade.workflow](diagrams/at-spatten-cascade.workflow.html) | showcase | receipt-verified |
| [at-triton-vs-cutedsl.workflow](diagrams/at-triton-vs-cutedsl.workflow.html) | showcase | receipt-verified |
| [cl-atom-generation-layouts.workflow](diagrams/cl-atom-generation-layouts.workflow.html) | showcase | receipt-verified |
| [cl-blis-formula-breaks.workflow](diagrams/cl-blis-formula-breaks.workflow.html) | showcase | receipt-verified |
| [cl-build-decision.workflow](diagrams/cl-build-decision.workflow.html) | showcase | receipt-verified |
| [cl-clc-vs-static.workflow](diagrams/cl-clc-vs-static.workflow.html) | showcase | receipt-verified |
| [cl-collective-dispatch-policy.workflow](diagrams/cl-collective-dispatch-policy.workflow.html) | showcase | receipt-verified |
| [cl-epilogue-api-generations.workflow](diagrams/cl-epilogue-api-generations.workflow.html) | showcase | receipt-verified |
| [cl-epilogue-fusion-io.workflow](diagrams/cl-epilogue-fusion-io.workflow.html) | showcase | receipt-verified |
| [cl-epilogue-standard-form.workflow](diagrams/cl-epilogue-standard-form.workflow.html) | showcase | receipt-verified |
| [cl-grouped-scheduler.workflow](diagrams/cl-grouped-scheduler.workflow.html) | showcase | receipt-verified |
| [cl-mma-atom-traits.workflow](diagrams/cl-mma-atom-traits.workflow.html) | showcase | receipt-verified |
| [cl-raster-order-l2.workflow](diagrams/cl-raster-order-l2.workflow.html) | showcase | receipt-verified |
| [cl-scheduler-persistent-vs-naive.workflow](diagrams/cl-scheduler-persistent-vs-naive.workflow.html) | showcase | receipt-verified |
| [cl-softmax-vs-layernorm.workflow](diagrams/cl-softmax-vs-layernorm.workflow.html) | showcase | receipt-verified |
| [cl-splitk-vs-streamk.workflow](diagrams/cl-splitk-vs-streamk.workflow.html) | showcase | receipt-verified |
| [cl-streamk-vs-tile.workflow](diagrams/cl-streamk-vs-tile.workflow.html) | showcase | receipt-verified |
| [cl-tcgen05-2cta.workflow](diagrams/cl-tcgen05-2cta.workflow.html) | showcase | receipt-verified |
| [cl-tcgen05-vs-wgmma.workflow](diagrams/cl-tcgen05-vs-wgmma.workflow.html) | showcase | receipt-verified |
| [cl-three-level-tiling.workflow](diagrams/cl-three-level-tiling.workflow.html) | showcase | receipt-verified |
| [cl-tiledmma-partition.workflow](diagrams/cl-tiledmma-partition.workflow.html) | showcase | receipt-verified |
| [cl-tiledmma-three-knobs.workflow](diagrams/cl-tiledmma-three-knobs.workflow.html) | showcase | receipt-verified |
| [cl-warp-specialization.workflow](diagrams/cl-warp-specialization.workflow.html) | showcase | receipt-verified |
| [cl-wgmma-sync.workflow](diagrams/cl-wgmma-sync.workflow.html) | showcase | receipt-verified |
| [cl-wgmma-vs-mma.workflow](diagrams/cl-wgmma-vs-mma.workflow.html) | showcase | receipt-verified |
| [cv-detection-core-difficulty.workflow](diagrams/cv-detection-core-difficulty.workflow.html) | showcase | receipt-verified |
| [cv-detr-set-prediction.workflow](diagrams/cv-detr-set-prediction.workflow.html) | showcase | receipt-verified |
| [cv-export-decisions.workflow](diagrams/cv-export-decisions.workflow.html) | showcase | receipt-verified |
| [cv-sam-three-parts.workflow](diagrams/cv-sam-three-parts.workflow.html) | showcase | receipt-verified |
| [cv-three-paradigms-inference.workflow](diagrams/cv-three-paradigms-inference.workflow.html) | showcase | receipt-verified |
| [cv-two-routes.workflow](diagrams/cv-two-routes.workflow.html) | showcase | receipt-verified |
| [cv-vit-deit.workflow](diagrams/cv-vit-deit.workflow.html) | showcase | receipt-verified |
| [cv-vit-why-data.workflow](diagrams/cv-vit-why-data.workflow.html) | showcase | receipt-verified |
| [deepgemm-ascend.workflow](diagrams/deepgemm-ascend.workflow.html) | unknown | legacy-unverified |
| [di-dg-grouped-gemm.workflow](diagrams/di-dg-grouped-gemm.workflow.html) | showcase | receipt-verified |
| [di-dg-mega-moe.workflow](diagrams/di-dg-mega-moe.workflow.html) | showcase | receipt-verified |
| [di-dg-sparse-mqa.workflow](diagrams/di-dg-sparse-mqa.workflow.html) | showcase | receipt-verified |
| [di-dspark-varlen.workflow](diagrams/di-dspark-varlen.workflow.html) | showcase | receipt-verified |
| [di-mhc-gain-runaway.workflow](diagrams/di-mhc-gain-runaway.workflow.html) | showcase | receipt-verified |
| [di-mhc-io-traffic.workflow](diagrams/di-mhc-io-traffic.workflow.html) | showcase | receipt-verified |
| [di-mhc-warpgroup-roles.workflow](diagrams/di-mhc-warpgroup-roles.workflow.html) | showcase | receipt-verified |
| [di-normal-voting.workflow](diagrams/di-normal-voting.workflow.html) | showcase | receipt-verified |
| [di-swa-bounded-replay.workflow](diagrams/di-swa-bounded-replay.workflow.html) | showcase | receipt-verified |
| [di-v3-deepep-dispatch.workflow](diagrams/di-v3-deepep-dispatch.workflow.html) | showcase | receipt-verified |
| [dm-esft-param-buffer.workflow](diagrams/dm-esft-param-buffer.workflow.html) | showcase | receipt-verified |
| [dm-esft-route-position.workflow](diagrams/dm-esft-route-position.workflow.html) | showcase | receipt-verified |
| [dm-esft-three-stages.workflow](diagrams/dm-esft-three-stages.workflow.html) | showcase | receipt-verified |
| [dm-esft-top-p.workflow](diagrams/dm-esft-top-p.workflow.html) | showcase | receipt-verified |
| [dm-esft-why-waste.workflow](diagrams/dm-esft-why-waste.workflow.html) | showcase | receipt-verified |
| [dm-r1-distill-vs-rl.workflow](diagrams/dm-r1-distill-vs-rl.workflow.html) | showcase | receipt-verified |
| [dm-r1-grpo-step.workflow](diagrams/dm-r1-grpo-step.workflow.html) | showcase | receipt-verified |
| [dv-ocr-compression-curve.workflow](diagrams/dv-ocr-compression-curve.workflow.html) | showcase | receipt-verified |
| [dv-ocr-deploy-path.workflow](diagrams/dv-ocr-deploy-path.workflow.html) | showcase | receipt-verified |
| [dv-ocr2-causal-flow.workflow](diagrams/dv-ocr2-causal-flow.workflow.html) | showcase | receipt-verified |
| [dv-vl2-dynamic-tiling.workflow](diagrams/dv-vl2-dynamic-tiling.workflow.html) | showcase | receipt-verified |
| [fa-backward-recompute.workflow](diagrams/fa-backward-recompute.workflow.html) | showcase | receipt-verified |
| [fa-fa1-tiling.workflow](diagrams/fa-fa1-tiling.workflow.html) | showcase | receipt-verified |
| [fa-fa2-defer-divide.workflow](diagrams/fa-fa2-defer-divide.workflow.html) | showcase | receipt-verified |
| [fa-online-softmax.workflow](diagrams/fa-online-softmax.workflow.html) | showcase | receipt-verified |
| [fa-splitq-vs-splitkv.workflow](diagrams/fa-splitq-vs-splitkv.workflow.html) | showcase | receipt-verified |
| [fa-why-not-tiling.workflow](diagrams/fa-why-not-tiling.workflow.html) | showcase | receipt-verified |
| [ma-ctx-picker.workflow](diagrams/ma-ctx-picker.workflow.html) | showcase | receipt-verified |
| [ma-ds-decouple-rope.workflow](diagrams/ma-ds-decouple-rope.workflow.html) | showcase | receipt-verified |
| [ma-ds-gen-stack.workflow](diagrams/ma-ds-gen-stack.workflow.html) | showcase | receipt-verified |
| [ma-ds-moe-route.workflow](diagrams/ma-ds-moe-route.workflow.html) | showcase | receipt-verified |
| [ma-ds-v2-group-route.workflow](diagrams/ma-ds-v2-group-route.workflow.html) | showcase | receipt-verified |
| [ma-dsa-pipeline.workflow](diagrams/ma-dsa-pipeline.workflow.html) | showcase | receipt-verified |
| [ma-gqa-uptraining.workflow](diagrams/ma-gqa-uptraining.workflow.html) | showcase | receipt-verified |
| [ma-kv-cache-bandwidth.workflow](diagrams/ma-kv-cache-bandwidth.workflow.html) | showcase | receipt-verified |
| [ma-llm-norm-lineage.workflow](diagrams/ma-llm-norm-lineage.workflow.html) | showcase | receipt-verified |
| [ma-llm-positional-encoding.workflow](diagrams/ma-llm-positional-encoding.workflow.html) | showcase | receipt-verified |
| [ma-llm-swiglu-ffn.workflow](diagrams/ma-llm-swiglu-ffn.workflow.html) | showcase | receipt-verified |
| [ma-llm-token-budget.workflow](diagrams/ma-llm-token-budget.workflow.html) | showcase | receipt-verified |
| [ma-mla-two-paths.workflow](diagrams/ma-mla-two-paths.workflow.html) | showcase | receipt-verified |
| [ma-moe-accounting.workflow](diagrams/ma-moe-accounting.workflow.html) | showcase | receipt-verified |
| [ma-open-model-choice.workflow](diagrams/ma-open-model-choice.workflow.html) | showcase | receipt-verified |
| [ma-rope-fail-mode.workflow](diagrams/ma-rope-fail-mode.workflow.html) | showcase | receipt-verified |
| [ma-v3-mla-absorb.workflow](diagrams/ma-v3-mla-absorb.workflow.html) | showcase | receipt-verified |
| [ma-v3-stack.workflow](diagrams/ma-v3-stack.workflow.html) | showcase | receipt-verified |
| [ma-v32-stack.workflow](diagrams/ma-v32-stack.workflow.html) | showcase | receipt-verified |
| [ma-yarn-by-parts.workflow](diagrams/ma-yarn-by-parts.workflow.html) | showcase | receipt-verified |
| [qt-aqlm-additive.workflow](diagrams/qt-aqlm-additive.workflow.html) | showcase | receipt-verified |
| [qt-ep-ptq-21-prs.workflow](diagrams/qt-ep-ptq-21-prs.workflow.html) | showcase | receipt-verified |
| [qt-ep-ptq-eight-details.workflow](diagrams/qt-ep-ptq-eight-details.workflow.html) | showcase | receipt-verified |
| [qt-ep-ptq-four-steps.workflow](diagrams/qt-ep-ptq-four-steps.workflow.html) | showcase | receipt-verified |
| [qt-ep-ptq-three-routes.workflow](diagrams/qt-ep-ptq-three-routes.workflow.html) | showcase | receipt-verified |
| [qt-fakequant-mbridge-export.workflow](diagrams/qt-fakequant-mbridge-export.workflow.html) | showcase | receipt-verified |
| [qt-fakequant-sidecar-restore.workflow](diagrams/qt-fakequant-sidecar-restore.workflow.html) | showcase | receipt-verified |
| [qt-fakequant-three-gates.workflow](diagrams/qt-fakequant-three-gates.workflow.html) | showcase | receipt-verified |
| [qt-minitron-distill-loss.workflow](diagrams/qt-minitron-distill-loss.workflow.html) | showcase | receipt-verified |
| [qt-minitron-importance-scoring.workflow](diagrams/qt-minitron-importance-scoring.workflow.html) | showcase | receipt-verified |
| [qt-minitron-pipeline.workflow](diagrams/qt-minitron-pipeline.workflow.html) | showcase | receipt-verified |
| [qt-moe-gptq-layerwise-loop.workflow](diagrams/qt-moe-gptq-layerwise-loop.workflow.html) | showcase | receipt-verified |
| [qt-moe-gptq-per-expert-hessian.workflow](diagrams/qt-moe-gptq-per-expert-hessian.workflow.html) | showcase | receipt-verified |
| [qt-moe-gptq-three-waits.workflow](diagrams/qt-moe-gptq-three-waits.workflow.html) | showcase | receipt-verified |
| [qt-puzzletron-hetero-mip.workflow](diagrams/qt-puzzletron-hetero-mip.workflow.html) | showcase | receipt-verified |
| [qt-two-bit-outlier-routes.workflow](diagrams/qt-two-bit-outlier-routes.workflow.html) | showcase | receipt-verified |
| [rng-curand-state-reuse.workflow](diagrams/rng-curand-state-reuse.workflow.html) | showcase | receipt-verified |
| [rng-mtgp-recursion.workflow](diagrams/rng-mtgp-recursion.workflow.html) | showcase | receipt-verified |
| [rng-philox-round.workflow](diagrams/rng-philox-round.workflow.html) | showcase | receipt-verified |
| [rng-picker.workflow](diagrams/rng-picker.workflow.html) | showcase | receipt-verified |
| [vm-aibrix-autoscale.workflow](diagrams/vm-aibrix-autoscale.workflow.html) | showcase | receipt-verified |
| [vm-aibrix-controlplane.workflow](diagrams/vm-aibrix-controlplane.workflow.html) | showcase | receipt-verified |
| [vm-aibrix-pd-storm.workflow](diagrams/vm-aibrix-pd-storm.workflow.html) | showcase | receipt-verified |
| [vm-aibrix-routing-algo.workflow](diagrams/vm-aibrix-routing-algo.workflow.html) | showcase | receipt-verified |
| [vm-async-snapshot.workflow](diagrams/vm-async-snapshot.workflow.html) | showcase | receipt-verified |
| [vm-attn-dispatch-chain.workflow](diagrams/vm-attn-dispatch-chain.workflow.html) | showcase | receipt-verified |
| [vm-attn-selection-pitfalls.workflow](diagrams/vm-attn-selection-pitfalls.workflow.html) | showcase | receipt-verified |
| [vm-attn-three-classes.workflow](diagrams/vm-attn-three-classes.workflow.html) | showcase | receipt-verified |
| [vm-awq-batch-heuristic.workflow](diagrams/vm-awq-batch-heuristic.workflow.html) | showcase | receipt-verified |
| [vm-cudagraph-dispatch.workflow](diagrams/vm-cudagraph-dispatch.workflow.html) | showcase | receipt-verified |
| [vm-fp8-quantkey.workflow](diagrams/vm-fp8-quantkey.workflow.html) | showcase | receipt-verified |
| [vm-guidellm-capacity-plan.workflow](diagrams/vm-guidellm-capacity-plan.workflow.html) | showcase | receipt-verified |
| [vm-guidellm-profiles.workflow](diagrams/vm-guidellm-profiles.workflow.html) | showcase | receipt-verified |
| [vm-guidellm-run-flow.workflow](diagrams/vm-guidellm-run-flow.workflow.html) | showcase | receipt-verified |
| [vm-gumbel-sampling.workflow](diagrams/vm-gumbel-sampling.workflow.html) | showcase | receipt-verified |
| [vm-hwp-entrypoints.workflow](diagrams/vm-hwp-entrypoints.workflow.html) | showcase | receipt-verified |
| [vm-hwp-load-path.workflow](diagrams/vm-hwp-load-path.workflow.html) | showcase | receipt-verified |
| [vm-hwp-responsibility.workflow](diagrams/vm-hwp-responsibility.workflow.html) | showcase | receipt-verified |
| [vm-hwp-worker-dispatch.workflow](diagrams/vm-hwp-worker-dispatch.workflow.html) | showcase | receipt-verified |
| [vm-llmc-pipeline-match.workflow](diagrams/vm-llmc-pipeline-match.workflow.html) | showcase | receipt-verified |
| [vm-llmc-three-abstractions.workflow](diagrams/vm-llmc-three-abstractions.workflow.html) | showcase | receipt-verified |
| [vm-omni-fullduplex-session.workflow](diagrams/vm-omni-fullduplex-session.workflow.html) | showcase | receipt-verified |
| [vm-omni-stage-pipeline.workflow](diagrams/vm-omni-stage-pipeline.workflow.html) | showcase | receipt-verified |
| [vm-pd-decision.workflow](diagrams/vm-pd-decision.workflow.html) | showcase | receipt-verified |
| [vm-penalty-math.workflow](diagrams/vm-penalty-math.workflow.html) | showcase | receipt-verified |
| [vm-persistent-batch.workflow](diagrams/vm-persistent-batch.workflow.html) | showcase | receipt-verified |
| [vm-prodstack-dashboard.workflow](diagrams/vm-prodstack-dashboard.workflow.html) | showcase | receipt-verified |
| [vm-prodstack-migration.workflow](diagrams/vm-prodstack-migration.workflow.html) | showcase | receipt-verified |
| [vm-prodstack-pd-disagg.workflow](diagrams/vm-prodstack-pd-disagg.workflow.html) | showcase | receipt-verified |
| [vm-prodstack-topology.workflow](diagrams/vm-prodstack-topology.workflow.html) | showcase | receipt-verified |
| [vm-quant-dispatch-chain.workflow](diagrams/vm-quant-dispatch-chain.workflow.html) | showcase | receipt-verified |
| [vm-ray-commgroup-init.workflow](diagrams/vm-ray-commgroup-init.workflow.html) | showcase | receipt-verified |
| [vm-ray-executor-compare.workflow](diagrams/vm-ray-executor-compare.workflow.html) | showcase | receipt-verified |
| [vm-ray-executor-dispatch.workflow](diagrams/vm-ray-executor-dispatch.workflow.html) | showcase | receipt-verified |
| [vm-ray-rank-rerank.workflow](diagrams/vm-ray-rank-rerank.workflow.html) | showcase | receipt-verified |
| [vm-router-consistenthash.workflow](diagrams/vm-router-consistenthash.workflow.html) | showcase | receipt-verified |
| [vm-router-request-path.workflow](diagrams/vm-router-request-path.workflow.html) | showcase | receipt-verified |
| [vm-router-strategy-tree.workflow](diagrams/vm-router-strategy-tree.workflow.html) | showcase | receipt-verified |
| [vm-sampler-order.workflow](diagrams/vm-sampler-order.workflow.html) | showcase | receipt-verified |
| [vm-sampling-params-fallback.workflow](diagrams/vm-sampling-params-fallback.workflow.html) | showcase | receipt-verified |
| [vm-scheduler-two-queues.workflow](diagrams/vm-scheduler-two-queues.workflow.html) | showcase | receipt-verified |
| [vm-spec-draft-verify.workflow](diagrams/vm-spec-draft-verify.workflow.html) | showcase | receipt-verified |
| [vm-spec-dynamic-adaptive.workflow](diagrams/vm-spec-dynamic-adaptive.workflow.html) | showcase | receipt-verified |
| [vm-spec-family.workflow](diagrams/vm-spec-family.workflow.html) | showcase | receipt-verified |
| [vm-spec-five-routes.workflow](diagrams/vm-spec-five-routes.workflow.html) | showcase | receipt-verified |
| [vm-spec-loss-target.workflow](diagrams/vm-spec-loss-target.workflow.html) | showcase | receipt-verified |
| [vm-spec-online-vs-offline.workflow](diagrams/vm-spec-online-vs-offline.workflow.html) | showcase | receipt-verified |
| [vm-spec-pipeline.workflow](diagrams/vm-spec-pipeline.workflow.html) | showcase | receipt-verified |
| [vm-sr-boolean-tree.workflow](diagrams/vm-sr-boolean-tree.workflow.html) | showcase | receipt-verified |
| [vm-sr-deployment.workflow](diagrams/vm-sr-deployment.workflow.html) | showcase | receipt-verified |
| [vm-sr-three-layer.workflow](diagrams/vm-sr-three-layer.workflow.html) | showcase | receipt-verified |
| [vm-sync-free-sampling.workflow](diagrams/vm-sync-free-sampling.workflow.html) | showcase | receipt-verified |
| [vm-v1-init-chain.workflow](diagrams/vm-v1-init-chain.workflow.html) | showcase | receipt-verified |
| [vm-vime-architecture.workflow](diagrams/vm-vime-architecture.workflow.html) | showcase | receipt-verified |
| [vm-vime-sync-vs-async.workflow](diagrams/vm-vime-sync-vs-async.workflow.html) | showcase | receipt-verified |

## Lifecycle (9)

| Diagram | Quality | Validation |
|---|---|---|
| [cl-arch-suffix.lifecycle](diagrams/cl-arch-suffix.lifecycle.html) | showcase | receipt-verified |
| [cl-scheduler-decision.lifecycle](diagrams/cl-scheduler-decision.lifecycle.html) | showcase | receipt-verified |
| [dm-r1-four-stages.lifecycle](diagrams/dm-r1-four-stages.lifecycle.html) | showcase | receipt-verified |
| [ma-v3-auxfree.lifecycle](diagrams/ma-v3-auxfree.lifecycle.html) | showcase | receipt-verified |
| [vm-aibrix-lora-adapter.lifecycle](diagrams/vm-aibrix-lora-adapter.lifecycle.html) | showcase | receipt-verified |
| [vm-llmc-oneshot-lifecycle.lifecycle](diagrams/vm-llmc-oneshot-lifecycle.lifecycle.html) | showcase | receipt-verified |
| [vm-pd-request-states.lifecycle](diagrams/vm-pd-request-states.lifecycle.html) | showcase | receipt-verified |
| [vm-request-lifecycle.lifecycle](diagrams/vm-request-lifecycle.lifecycle.html) | showcase | receipt-verified |
| [vm-router-circuit-breaker.lifecycle](diagrams/vm-router-circuit-breaker.lifecycle.html) | showcase | receipt-verified |

## Timeline (6)

| Diagram | Quality | Validation |
|---|---|---|
| [dm-r1-lineage.timeline](diagrams/dm-r1-lineage.timeline.html) | showcase | receipt-verified |
| [dv-vl-to-vl2-evolution.timeline](diagrams/dv-vl-to-vl2-evolution.timeline.html) | showcase | receipt-verified |
| [ma-ctx-timeline.timeline](diagrams/ma-ctx-timeline.timeline.html) | showcase | receipt-verified |
| [ma-head-sharing-lineage.timeline](diagrams/ma-head-sharing-lineage.timeline.html) | showcase | receipt-verified |
| [ma-llm-evolution.timeline](diagrams/ma-llm-evolution.timeline.html) | showcase | receipt-verified |
| [ma-open-model-timeline.timeline](diagrams/ma-open-model-timeline.timeline.html) | showcase | receipt-verified |

## Waterfall (7)

| Diagram | Quality | Validation |
|---|---|---|
| [cl-multistage-pipeline.waterfall](diagrams/cl-multistage-pipeline.waterfall.html) | showcase | receipt-verified |
| [di-v3-a2a-timeline.waterfall](diagrams/di-v3-a2a-timeline.waterfall.html) | showcase | receipt-verified |
| [di-v3-dualpipe-schedule.waterfall](diagrams/di-v3-dualpipe-schedule.waterfall.html) | showcase | receipt-verified |
| [di-zos-overlap.waterfall](diagrams/di-zos-overlap.waterfall.html) | showcase | receipt-verified |
| [fa-fa3-async-pipeline.waterfall](diagrams/fa-fa3-async-pipeline.waterfall.html) | showcase | receipt-verified |
| [vm-chunked-prefill-timeline.waterfall](diagrams/vm-chunked-prefill-timeline.waterfall.html) | showcase | receipt-verified |
| [vm-spec-timeline.waterfall](diagrams/vm-spec-timeline.waterfall.html) | showcase | receipt-verified |

## Tree (30)

| Diagram | Quality | Validation |
|---|---|---|
| [at-fa4-limits.tree](diagrams/at-fa4-limits.tree.html) | showcase | receipt-verified |
| [at-laser-spatten-longtail.tree](diagrams/at-laser-spatten-longtail.tree.html) | showcase | receipt-verified |
| [cl-build-pitfalls.tree](diagrams/cl-build-pitfalls.tree.html) | showcase | receipt-verified |
| [cl-gmma-layout-atoms.tree](diagrams/cl-gmma-layout-atoms.tree.html) | showcase | receipt-verified |
| [cl-layout-algebra.tree](diagrams/cl-layout-algebra.tree.html) | showcase | receipt-verified |
| [cl-tcgen05-atom-naming.tree](diagrams/cl-tcgen05-atom-naming.tree.html) | showcase | receipt-verified |
| [cl-tiled-mma-layers.tree](diagrams/cl-tiled-mma-layers.tree.html) | showcase | receipt-verified |
| [cl-two-level-microkernel.tree](diagrams/cl-two-level-microkernel.tree.html) | showcase | receipt-verified |
| [cl-wgmma-value.tree](diagrams/cl-wgmma-value.tree.html) | showcase | receipt-verified |
| [cv-engine-manifest.tree](diagrams/cv-engine-manifest.tree.html) | showcase | receipt-verified |
| [di-v3-memory-budget.tree](diagrams/di-v3-memory-budget.tree.html) | showcase | receipt-verified |
| [dm-r1-reward-choices.tree](diagrams/dm-r1-reward-choices.tree.html) | showcase | receipt-verified |
| [dv-ocr-two-gen.tree](diagrams/dv-ocr-two-gen.tree.html) | showcase | receipt-verified |
| [dv-vl2-moe-variants.tree](diagrams/dv-vl2-moe-variants.tree.html) | showcase | receipt-verified |
| [ma-ctx-unified-map.tree](diagrams/ma-ctx-unified-map.tree.html) | showcase | receipt-verified |
| [ma-open-model-lineage.tree](diagrams/ma-open-model-lineage.tree.html) | showcase | receipt-verified |
| [ma-v3-vs-v2-diff.tree](diagrams/ma-v3-vs-v2-diff.tree.html) | showcase | receipt-verified |
| [ma-v32-diff.tree](diagrams/ma-v32-diff.tree.html) | showcase | receipt-verified |
| [qt-ep-ptq-register-transformers.tree](diagrams/qt-ep-ptq-register-transformers.tree.html) | showcase | receipt-verified |
| [qt-fakequant-nll-diff.tree](diagrams/qt-fakequant-nll-diff.tree.html) | showcase | receipt-verified |
| [qt-minitron-mamba-state-consistency.tree](diagrams/qt-minitron-mamba-state-consistency.tree.html) | showcase | receipt-verified |
| [qt-moe-gptq-batch-evolution.tree](diagrams/qt-moe-gptq-batch-evolution.tree.html) | showcase | receipt-verified |
| [vm-attn-backend-enum.tree](diagrams/vm-attn-backend-enum.tree.html) | showcase | receipt-verified |
| [vm-guidellm-metrics.tree](diagrams/vm-guidellm-metrics.tree.html) | showcase | receipt-verified |
| [vm-llmc-precision-split.tree](diagrams/vm-llmc-precision-split.tree.html) | showcase | receipt-verified |
| [vm-omni-inheritance.tree](diagrams/vm-omni-inheritance.tree.html) | showcase | receipt-verified |
| [vm-sr-signal-types.tree](diagrams/vm-sr-signal-types.tree.html) | showcase | receipt-verified |
| [vm-token-budget-model.tree](diagrams/vm-token-budget-model.tree.html) | showcase | receipt-verified |
| [vm-v1-executor-choice.tree](diagrams/vm-v1-executor-choice.tree.html) | showcase | receipt-verified |
| [vm-vime-weight-sync.tree](diagrams/vm-vime-weight-sync.tree.html) | showcase | receipt-verified |

## Totals

- Diagrams: 286
- Legacy unverified: 4
- Failures: 0
