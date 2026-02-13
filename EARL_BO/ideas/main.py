from EARL_BO.ideas.experiment import ExperimentConfig, ExperimentRunner

if __name__ == '__main__':
    horizons = [1]
    for horizon in horizons:
        config = ExperimentConfig(
            dimension=1,
            test_func_name='ackley',
            num_runs=1,
            horizon=horizon
        )
        runner = ExperimentRunner(config)
        runner.run_all()