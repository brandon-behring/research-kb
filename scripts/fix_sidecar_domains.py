#!/usr/bin/env python3
"""Fix domain misattributions in migrated sidecar JSON files.

Phase N: Domain Gap Expansion — corrects legacy domain labels
(other, programming, ml_stats, math, nlp, causal) to valid domain IDs
based on content-aware analysis of each book.

Usage:
    python scripts/fix_sidecar_domains.py          # Dry run (default)
    python scripts/fix_sidecar_domains.py --apply   # Apply changes
"""

import argparse
import json
from pathlib import Path

# Content-aware correction map: filename -> correct domain
# Built from full audit of all 95 migrated/ sidecar JSONs
CORRECTIONS: dict[str, str] = {
    "1_arnaud_lauret_-_the_design_of_web_apis_0_-_libge_nd.json": "software_engineering",
    "aditya_bhargava_grokking_algorithms_an_illustrated_nd.json": "algorithms",
    "aileen_nielsen_-_practical_time_series_analysis__p_nd.json": "time_series",
    "alexander_s_kulikov_and_pavel_pevzner_-_learning_a_nd.json": "algorithms",
    "andrew_c_fry__vladimir_m_zatsiorsky__william_j_kra_nd.json": "fitness",
    "anthony_scopatz_kathryn_d_huff_-_effective_computa_nd.json": "software_engineering",
    "biostatmethods_nd.json": "statistics",
    "bronson_r_costa_gb_schaums_outlines_differential_e_nd.json": "mathematics",
    "causal_inference_for_data_science_nd.json": "causal_inference",
    "charu_c_aggarwal_-_linear_algebra_and_optimization_nd.json": "mathematics",
    "chris_fregly_antje_barth_-_data_science_on_aws__im_nd.json": "ml_engineering",
    "class_health_-_strength_training__practical_progra_nd.json": "fitness",
    "classroom_resource_materials_david_a_b_miller_-_qu_nd.json": "mathematics",
    "computer_age_statistical_inference_nd.json": "statistics",
    "data_analysis_with_python_and_pyspark_v13_meap_nd.json": "data_science",
    "david_c_lay_steven_r_lay_judi_j_mcdonald_linear_al_nd.json": "mathematics",
    "david_j_griffiths_darrell_f_schroeter_introduction_nd.json": "mathematics",
    "david_m_diez_christopher_d_barr_mine_çetinkaya-run_nd.json": "statistics",
    "deep_learning_with_python_second_editio_nd.json": "deep_learning",
    "deep_learning_with_python_second_editio_v7_nd.json": "deep_learning",
    "deep_learning_with_pytorch_nd.json": "deep_learning",
    "emmanuel_ameisen_-_building_machine_learning_power_nd.json": "ml_engineering",
    "ensemble_learning_algorithms_with_python_nd.json": "machine_learning",
    "exploring_functional_programming_nd.json": "functional_programming",
    "eyal_wirsansky_hands_on_genetic_algorithms_with_py_nd.json": "machine_learning",
    "functional_programming_in_scala_second__v2_meap_nd.json": "functional_programming",
    "get_programming_with_haskell_nd.json": "functional_programming",
    "get_programming_with_scala_nd.json": "functional_programming",
    "gilbert_strang_-_linear_algebra_and_its_applicatio_nd.json": "mathematics",
    "graduate_texts_in_mathematics_60_v_i_arnold_auth_-_nd.json": "mathematics",
    "grokking_algorithms_nd.json": "algorithms",
    "gtm267_quantum_theory_for_mathematicians_nd.json": "mathematics",
    "hannes_hapke_catherine_nelson_-_building_machine_l_nd.json": "ml_engineering",
    "haskell_in_depth_nd.json": "functional_programming",
    "herbert_goldstein__charles_p_poole__john_l_safko_-_nd.json": "mathematics",
    "imbalanced_classification_with_python_nd.json": "machine_learning",
    "inside_deep_learning_v9_meap_nd.json": "deep_learning",
    "jay_wengrow_a_common_sense_guide_to_data_structure_nd.json": "algorithms",
    "jj_geewax_-_api_design_patterns-manning_publicatio_nd.json": "software_engineering",
    "jj_geewax_-_google_cloud_platform_in_action-mannin_nd.json": "software_engineering",
    "j_j_sakurai__jim_napolitano_-_modern_quantum_mecha_nd.json": "mathematics",
    "john_hubbard_-_schaums_outline_of_programming_with_nd.json": "software_engineering",
    "jorge_v_josé_eugene_j_saletan_-_classical_dynamics_nd.json": "mathematics",
    "judea_pearl_madelyn_glymour_nicholas_p_jewell_-_ca_nd.json": "causal_inference",
    "laurence_moroney_-_ai_and_machine_learning_for_cod_nd.json": "machine_learning",
    "lee_peter_-_bayesian_statistics__an_introduction-j_nd.json": "statistics",
    "lipovača_miran_learn_you_a_haskell_for_great_good__nd.json": "functional_programming",
    "lloyd_n_trefethen_david_bau_iii_-_numerical_linear_nd.json": "mathematics",
    "machine_learning_algorithms_from_scratch_nd.json": "machine_learning",
    "machine_learning_control_taming_nonlinear_dynamics_nd.json": "machine_learning",
    "machine_learning_mastery_with_python_nd.json": "machine_learning",
    "marc_peter_deisenroth__a_aldo_faisal__cheng_soon_o_nd.json": "mathematics",
    "mathprogrammingintro_nd.json": "mathematics",
    "miklavcic_sj_-_an_illustrative_guide_to_multivaria_nd.json": "mathematics",
    "miller_bn_ranum_dl_-_problem_solving_with_algorith_nd.json": "algorithms",
    "monographs_on_mathematical_modeling_and_computatio_nd.json": "mathematics",
    "murtaza_haider_-_getting_started_with_data_science_nd.json": "data_science",
    "mykel_j_kochenderfer_tim_a_wheeler_-_algorithms_fo_nd.json": "algorithms",
    "natural_language_processing_in_action_1_nd.json": "rag_llm",
    "nick_alteen_jennifer_fisher_casey_gerena_wes_gruve_nd.json": "software_engineering",
    "openintro-statistics_nd.json": "statistics",
    "optimization_for_machine_learning_nd.json": "mathematics",
    "practices_of_the_python_pro_nd.json": "software_engineering",
    "publishing_python_packages_nd.json": "software_engineering",
    "python_workout_nd.json": "software_engineering",
    "quantitative_economics_with_julia_nd.json": "economics",
    "quantitative_methodology_joop_j_hox_mirjam_moerbee_nd.json": "statistics",
    "quantum_field_theory_for_the_gifted_amateur-oxford_nd.json": "mathematics",
    "robert_d_klauber_-_student_friendly_quantum_field__nd.json": "mathematics",
    "roughgarden_t_-_algorithms_illuminated_part_3_gree_nd.json": "algorithms",
    "ryan_t_white_archana_tikayat_ray_-_practical_discr_nd.json": "mathematics",
    "santanu_pattanayak_auth_-_pro_deep_learning_with_t_nd.json": "deep_learning",
    "scalabyexample_nd.json": "functional_programming",
    "schaums_outline_series_j_r_hubbard_-_schaums_outli_nd.json": "software_engineering",
    "schaums_outlines_seymour_lipschutz_marc_lipson_lin_nd.json": "mathematics",
    "shalev-shwartz_s_ben-david_s_-_understanding_machi_nd.json": "machine_learning",
    "skills_of_a_software_developer_v3_meap_nd.json": "software_engineering",
    "skills_of_a_successful_software_engineer_nd.json": "software_engineering",
    "software_environments_tools_lloyd_n_trefethen_-_sp_nd.json": "mathematics",
    "springer_texts_in_statistics_larry_wasserman_-_all_nd.json": "statistics",
    "springer_texts_in_statistics_robert_h_shumway_davi_nd.json": "time_series",
    "springer_theses_vincent_traag_auth_-_algorithms_an_nd.json": "data_science",
    "statistical_methods_for_machine_learning_nd.json": "statistics",
    "stefan_thurner_rudolf_hanel_peter_klimek_-_introdu_nd.json": "mathematics",
    "stephen_boyd_lieven_vandenberghe_-_introduction_to_nd.json": "mathematics",
    "steven_f_railsback__volker_grimm_-_agent-based_and_nd.json": "data_science",
    "studies_in_systems_decision_and_control_103_li_ben_nd.json": "machine_learning",
    "tensorflow_20_in_action_v12_meap_nd.json": "deep_learning",
    "texts_in_applied_mathematics_carmen_chicone_-_ordi_nd.json": "mathematics",
    "therese_m_donovan__ruth_m_mickey_-_bayesian_statis_nd.json": "statistics",
    "tim_roughgarden_-_algorithms_illuminated__part_1___nd.json": "algorithms",
    "tim_roughgarden_-_algorithms_illuminated_part_2__g_nd.json": "algorithms",
    "trefethen_ln_-_finite_difference_and_spectral_meth_nd.json": "mathematics",
    "undergraduate_lecture_notes_in_physics_joshua_izaa_nd.json": "mathematics",
    "use_r_eric_d_kolaczyk_gábor_csárdi_-_statistical_a_nd.json": "statistics",
    "vitaly_bragilevsky_-_haskell_in_depth-manning_publ_nd.json": "functional_programming",
    "xgboost_with_python_nd.json": "machine_learning",
}


def main():
    parser = argparse.ArgumentParser(description="Fix sidecar domain misattributions.")
    parser.add_argument("--apply", action="store_true", help="Apply changes (default: dry run)")
    args = parser.parse_args()

    migrated = Path(__file__).parent.parent / "fixtures" / "textbooks" / "migrated"
    if not migrated.exists():
        print(f"Error: {migrated} does not exist")
        return

    fixed = 0
    skipped = 0
    errors = 0

    for filename, correct_domain in sorted(CORRECTIONS.items()):
        json_path = migrated / filename
        if not json_path.exists():
            print(f"  MISSING: {filename}")
            errors += 1
            continue

        with open(json_path) as f:
            data = json.load(f)

        current = data.get("domain", "NONE")
        if current == correct_domain:
            skipped += 1
            continue

        if args.apply:
            data["domain"] = correct_domain
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  FIXED: {filename}: {current} -> {correct_domain}")
        else:
            print(f"  WOULD FIX: {filename}: {current} -> {correct_domain}")
        fixed += 1

    action = "Fixed" if args.apply else "Would fix"
    print(f"\n{action}: {fixed} | Skipped (already correct): {skipped} | Missing: {errors}")
    if not args.apply and fixed > 0:
        print("Run with --apply to make changes.")


if __name__ == "__main__":
    main()
