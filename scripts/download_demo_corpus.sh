#!/bin/bash
# Download open-access arXiv papers for the demo corpus.
# These papers are freely available under arXiv's license.
#
# Usage: ./scripts/download_demo_corpus.sh [output_dir]

set -euo pipefail

OUTPUT_DIR="${1:-fixtures/demo/papers}"
mkdir -p "$OUTPUT_DIR"

echo "Downloading demo corpus to $OUTPUT_DIR..."

# -- Causal Inference (~12 papers) --

download() {
    local arxiv_id="$1"
    local filename="$2"
    local target="$OUTPUT_DIR/$filename"

    if [ -f "$target" ]; then
        echo "  [skip] $filename (already exists)"
        return 0
    fi

    echo "  [download] $filename (arXiv:$arxiv_id)"
    curl -sL "https://arxiv.org/pdf/$arxiv_id" -o "$target" || {
        echo "  [WARN] Failed to download $filename"
        return 0
    }
    # Be polite to arXiv servers
    sleep 3
}

echo ""
echo "=== Causal Inference ==="
download "1608.00060v7" "chernozhukov_dml_2018.pdf"
download "1510.04342v4" "wager_athey_causal_forests_2018.pdf"
download "1712.09988v3" "kunzel_metalearners_2019.pdf"
download "1504.01132v3" "athey_imbens_recursive_partitioning_2016.pdf"
download "1803.09015v6" "callaway_santanna_staggered_did_2021.pdf"
download "2108.02196v2" "abadie_zhao_synthetic_controls_2021.pdf"
download "1011.1079v2" "imai_keele_tingley_mediation_2010.pdf"
download "1607.00699v2" "athey_imbens_state_of_applied_econometrics_2017.pdf"
download "1903.10075v4" "athey_imbens_ml_methods_economists_2019.pdf"

echo ""
echo "=== RAG / LLM Foundations ==="
download "2005.11401v4" "lewis_rag_2020.pdf"
download "2312.10997v5" "gao_rag_survey_2024.pdf"
download "1706.03762v7" "vaswani_attention_2017.pdf"
download "2203.02155v1" "wei_chain_of_thought_2022.pdf"
download "2104.08691v2" "izacard_grave_passage_retrieval_2021.pdf"

echo ""
echo "=== Knowledge Graphs + LLMs ==="
download "2306.08302v2" "pan_unifying_llms_kgs_2024.pdf"
download "2404.16130v1" "edge_graphrag_2024.pdf"
download "2308.14522v1" "hu_empowering_llms_kgs_2023.pdf"

echo ""
echo "=== Additional Foundational Papers ==="
download "2005.14165v4" "brown_gpt3_2020.pdf"
download "2201.11903v6" "wei_chain_of_thought_prompting_2023.pdf"
download "2112.10752v3" "rombach_latent_diffusion_2022.pdf"
download "2307.09288v2" "touvron_llama2_2023.pdf"
download "1810.04805v2" "devlin_bert_2019.pdf"
download "2305.10601v3" "zheng_judging_llm_arena_2023.pdf"
download "2310.06825v4" "wang_self_knowledge_llms_2023.pdf"
download "2309.01219v2" "zhang_siren_call_2023.pdf"

echo ""
echo "=== Download Complete ==="
echo "Papers saved to: $OUTPUT_DIR"
echo "Total files: $(ls "$OUTPUT_DIR"/*.pdf 2>/dev/null | wc -l)"
