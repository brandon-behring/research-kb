# Time Series and Forecasting

**Module ID**: `07_time_series_and_forecasting`
**Prerequisites**: `03_computational_bayes`, `04_bayesian_workflow_and_model_checks`, `05_regression_and_glms`, `06_hierarchical_models`
**Goal**: Move from classical ARIMA foundations into state-space models, Bayesian time series, and probabilistic forecasting.

## Primary materials

- **Time Series Analysis and Its Applications**: Entry point for the time-series branch. (`fixtures/textbooks/migrated/springer_texts_in_statistics_robert_h_shumway_davi_nd.pdf`)
- **Time Series Analysis: Forecasting and Control**: Box-Jenkins reference. (`fixtures/textbooks/box_jenkins_time_series_2015.pdf`)
- **Time Series Analysis**: Deeper econometric time-series treatment. (`fixtures/textbooks/hamilton_time_series_analysis_1994.pdf`)
- **Time Series Analysis by State Space Methods**: State-space and Kalman backbone. (`fixtures/library_books/James Durbin, Siem Jan Koopman - Time Series Analysis by State Space Methods_ Second Edition (Oxford Statistical Science Series)-Oxford University Press (2012).pdf`)
- **Time Series: Modeling, Computation, and Inference**: Modern Bayesian time-series reference. (`fixtures/library_books/(Chapman & Hall_CRC Texts in Statistical Science) West, Mike_ Prado, Raquel - Time Series_ Modeling, Computation, and Inference-Chapman and Hall_CRC (2010).pdf`)
- **Bayesian Time Series Models**: Explicit Bayesian state-space and time-series resource. (`fixtures/library_books/Barber D., Cemgil A.T., Chiappa S. (eds.) - Bayesian Time Series Models-Cambridge University Press (2011).pdf`)
- **Applied Bayesian Forecasting and Time Series Analysis**: Pedagogically useful Bayesian forecasting text. (`fixtures/library_books/Andy Pole, Mike West, Jeff Harrison (auth.) - Applied Bayesian Forecasting and Time Series Analysis-Springer US (1994).pdf`)

## Support materials

- **Forecasting: Principles and Practice**: Free applied forecasting reference. (https://otexts.com/fpp3/)
- **M4 competition data**: Benchmark forecasting dataset and methods. (https://github.com/Mcompetitions/M4-methods)
- **Stan example models**: Secondary comparison stack and real data examples. (https://github.com/stan-dev/example-models)

## Artifacts

- `notebooks/`: two runnable notebook skeletons
- `exercises/worked.md`: worked prompts
- `exercises/unworked.md`: unworked prompts
- `case_study.md`: mini-project / case-study brief
- `references/`: primary, support, and optional reference lists

## Exercise sources

- Shumway/Stoffer examples
- BDA course traffic deaths and Kilpisjarvi data
- M4 subsets

## Checkpoint gate

- You can fit one ARIMA/state-space/Bayesian forecast model and evaluate calibration or forecast error on held-out data.
