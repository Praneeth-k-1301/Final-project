import unittest

import numpy as np
import pandas as pd

from benchmark_full_pipeline import (build_non_5050_strategy_plans,
                                     build_model_configs,
                                     build_preprocessor,
                                     build_strategy_comparison_table,
                                     class_distribution,
                                     collect_binary_metrics,
                                     determine_cv_folds,
                                     fit_params_for_training_strategy,
                                     normalize_training_strategy,
                                     prepare_strategy_dataset,
                                     resampled_distribution_for_reporting,
                                     sample_prevalence_faithful_subset,
                                     sample_case_control_subset,
                                     select_speed_accuracy_tradeoff,
                                     should_apply_smote,
                                     split_train_validation_test,
                                     tune_decision_threshold)


class BenchmarkPipelineHelpersTestCase(unittest.TestCase):
    def test_should_apply_smote_for_severe_imbalance(self):
        labels = pd.Series([0] * 8 + [1] * 2)
        self.assertTrue(should_apply_smote(labels))

    def test_class_distribution_reports_counts(self):
        summary = class_distribution(pd.Series([0, 0, 0, 1]))
        self.assertEqual(summary['counts'], {'0': 3, '1': 1})
        self.assertAlmostEqual(summary['minorityFraction'], 0.25)

    def test_build_preprocessor_handles_mixed_columns(self):
        frame = pd.DataFrame({'numeric_feature': [1.0, 2.0], 'category_feature': ['a', 'b']})
        preprocessor, numeric_columns, categorical_columns = build_preprocessor(frame)
        transformed = preprocessor.fit_transform(frame)
        self.assertEqual(numeric_columns, ['numeric_feature'])
        self.assertEqual(categorical_columns, ['category_feature'])
        self.assertEqual(transformed.shape[0], 2)

    def test_select_speed_accuracy_tradeoff_prefers_fast_and_strong_auc(self):
        table = pd.DataFrame([
            {'Model': 'slow_best', 'ROC-AUC': 0.95, 'Train_Time(s)': 120.0},
            {'Model': 'balanced_choice', 'ROC-AUC': 0.94, 'Train_Time(s)': 10.0},
            {'Model': 'fast_but_weaker', 'ROC-AUC': 0.88, 'Train_Time(s)': 2.0},
        ])
        self.assertEqual(select_speed_accuracy_tradeoff(table), 'balanced_choice')

    def test_sample_case_control_subset_balances_real_rows(self):
        features = pd.DataFrame({'feature': range(12)})
        labels = pd.Series([0] * 9 + [1] * 3)
        sampled_features, sampled_labels, metadata = sample_case_control_subset(features, labels)
        self.assertEqual(len(sampled_features), 6)
        self.assertEqual(sampled_labels.value_counts().to_dict(), {0: 3, 1: 3})
        self.assertEqual(metadata['strategy'], 'case_control')
        self.assertEqual(metadata['positiveRows'], 3)
        self.assertEqual(metadata['negativeRows'], 3)

    def test_sample_prevalence_faithful_subset_preserves_imbalance(self):
        features = pd.DataFrame({'feature': range(100)})
        labels = pd.Series([0] * 90 + [1] * 10)
        sampled_features, sampled_labels, metadata = sample_prevalence_faithful_subset(features, labels, max_rows=50)
        self.assertEqual(len(sampled_features), 50)
        self.assertEqual(sampled_labels.value_counts().to_dict(), {0: 45, 1: 5})
        self.assertEqual(metadata['strategy'], 'prevalence_faithful_stratified_subset')
        self.assertEqual(metadata['sourceRows'], 100)

    def test_split_train_validation_test_uses_70_15_15_and_preserves_positives(self):
        features = pd.DataFrame({'feature': range(100)})
        labels = pd.Series([0] * 90 + [1] * 10)
        splits = split_train_validation_test(features, labels)
        self.assertEqual(len(splits['train'][0]), 70)
        self.assertEqual(len(splits['validation'][0]), 15)
        self.assertEqual(len(splits['test'][0]), 15)
        self.assertGreaterEqual(int((splits['train'][1] == 1).sum()), 2)
        self.assertGreaterEqual(int((splits['validation'][1] == 1).sum()), 1)
        self.assertGreaterEqual(int((splits['test'][1] == 1).sum()), 1)

    def test_tune_decision_threshold_prefers_balanced_f1(self):
        labels = pd.Series([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.10, 0.30, 0.55, 0.60, 0.80, 0.90])
        threshold = tune_decision_threshold(labels, probabilities)
        self.assertGreaterEqual(threshold, 0.55)
        self.assertLessEqual(threshold, 0.60)

    def test_build_model_configs_for_prevalence_uses_requested_four_models(self):
        configs = build_model_configs(pd.Series([0] * 70 + [1] * 5), prevalence_faithful=True)
        self.assertEqual(set(configs.keys()), {
            'HistGradientBoostingClassifier',
            'RandomForestClassifier',
            'LightGBM',
            'XGBoost',
        })
        self.assertIn('classifier__scale_pos_weight', configs['LightGBM']['params'])
        self.assertIn('classifier__scale_pos_weight', configs['XGBoost']['params'])

    def test_prevalence_hgb_fit_params_skip_non_none_training_strategies(self):
        configs = build_model_configs(
            pd.Series([0] * 70 + [1] * 5),
            prevalence_faithful=True,
            training_strategy={'kind': 'random_oversample'},
        )
        builder = configs['HistGradientBoostingClassifier']['fit_params_builder']
        self.assertEqual(builder(pd.Series([0] * 70 + [1] * 5), {'kind': 'random_oversample'}), {})
        fit_params = builder(pd.Series([0] * 70 + [1] * 5), {'kind': 'none'})
        self.assertIn('classifier__sample_weight', fit_params)

    def test_normalize_training_strategy_defaults_unknown_kind_to_none(self):
        strategy = normalize_training_strategy({'kind': 'unsupported', 'name': 'custom'})
        self.assertEqual(strategy['kind'], 'none')
        self.assertEqual(strategy['name'], 'custom')
        self.assertIsNone(strategy['targetNegativeToPositiveRatio'])

    def test_fit_params_for_training_strategy_builds_balanced_sample_weights(self):
        labels = pd.Series([0, 0, 0, 1])
        fit_params = fit_params_for_training_strategy({'kind': 'sample_weight'}, labels)
        self.assertIn('classifier__sample_weight', fit_params)
        sample_weight = fit_params['classifier__sample_weight']
        self.assertEqual(len(sample_weight), 4)
        self.assertGreater(sample_weight[-1], sample_weight[0])

    def test_determine_cv_folds_adapts_to_minority_count(self):
        labels = pd.Series([0] * 100 + [1] * 12)
        self.assertEqual(determine_cv_folds(labels), 2)
        self.assertEqual(determine_cv_folds(labels, min_positives_per_fold=2), 5)

    def test_build_non_5050_strategy_plans_returns_expected_strategies(self):
        plans = build_non_5050_strategy_plans()
        self.assertEqual(len(plans), 7)
        self.assertEqual(plans[0]['name'], 'prevalence_10k_baseline')
        self.assertEqual(plans[-1]['trainingStrategy']['kind'], 'random_oversample')

    def test_prepare_strategy_dataset_supports_case_control_and_prevalence_subset(self):
        features = pd.DataFrame({'feature': range(30)})
        labels = pd.Series([0] * 24 + [1] * 6)
        cc_features, cc_labels, cc_meta = prepare_strategy_dataset(
            features,
            labels,
            {'kind': 'case_control', 'negativeToPositiveRatio': 2},
        )
        self.assertEqual(len(cc_features), 18)
        self.assertEqual(cc_labels.value_counts().to_dict(), {0: 12, 1: 6})
        self.assertEqual(cc_meta['strategy'], 'case_control')
        prev_features, prev_labels, prev_meta = prepare_strategy_dataset(
            features,
            labels,
            {'kind': 'prevalence_faithful', 'maxRows': 15},
        )
        self.assertEqual(len(prev_features), 15)
        self.assertEqual(prev_labels.value_counts().to_dict(), {0: 12, 1: 3})
        self.assertEqual(prev_meta['strategy'], 'prevalence_faithful_stratified_subset')

    def test_resampled_distribution_for_reporting_records_sample_weight_metadata(self):
        features = pd.DataFrame({'feature': range(20)})
        labels = pd.Series([0] * 15 + [1] * 5)
        before, after, metadata = resampled_distribution_for_reporting(
            features,
            labels,
            {'kind': 'sample_weight', 'name': 'balanced_sample_weight'},
        )
        self.assertEqual(before, after)
        self.assertEqual(metadata['trainingStrategy']['kind'], 'sample_weight')
        self.assertIn('sampleWeightByClass', metadata)
        self.assertGreater(metadata['sampleWeightByClass']['1'], metadata['sampleWeightByClass']['0'])

    def test_collect_binary_metrics_returns_nan_auc_for_single_class_labels(self):
        metrics = collect_binary_metrics(
            labels=[0, 0, 0],
            predictions=[0, 0, 0],
            probabilities=[0.1, 0.2, 0.3],
        )
        self.assertTrue(np.isnan(metrics['ROC-AUC']))
        self.assertIn('PR-AUC', metrics)
        self.assertEqual(metrics['PR-AUC'], 0.0)

    def test_build_strategy_comparison_table_uses_best_row_per_strategy(self):
        summary = {
            'evaluations': {
                'strategy_a': {
                    'paperOption': 'natural_imbalance',
                    'bestModel': 'ModelA',
                    'trainingStrategy': {'name': 'none'},
                    'cvFolds': 2,
                    'evaluationDesign': {'sampledRows': 100, 'positiveRows': 10, 'negativeRows': 90},
                    'comparisonTable': [
                        {'Model': 'ModelA', 'F1': 0.4, 'ROC-AUC': 0.7, 'Precision': 0.5, 'Recall': 0.35},
                        {'Model': 'ModelB', 'F1': 0.3, 'ROC-AUC': 0.8, 'Precision': 0.4, 'Recall': 0.25},
                    ],
                },
                'strategy_b': {
                    'paperOption': 'class_weight_only',
                    'bestModel': 'ModelC',
                    'trainingStrategy': {'name': 'balanced_sample_weight'},
                    'cvFolds': 5,
                    'evaluationDesign': {'sampledRows': 200, 'positiveRows': 20, 'negativeRows': 180},
                    'comparisonTable': [
                        {'Model': 'ModelC', 'F1': 0.5, 'ROC-AUC': 0.75, 'Precision': 0.55, 'Recall': 0.45},
                    ],
                },
            }
        }
        comparison = build_strategy_comparison_table(summary)
        self.assertEqual(list(comparison['Strategy']), ['strategy_b', 'strategy_a'])
        self.assertEqual(comparison.iloc[0]['Best_Model_Name'], 'ModelC')


if __name__ == '__main__':
    unittest.main()