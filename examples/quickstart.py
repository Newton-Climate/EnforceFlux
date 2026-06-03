from enforceflux.config import load_config
from enforceflux.osse import run_osse

if __name__ == "__main__":
    config = load_config("examples/quickstart_config.json")
    output = run_osse(config)

    print("Posterior mean:", output.inversion.x_posterior)
    print("Posterior std:", output.metrics.posterior_std)
    print("Fisher info:\n", output.inversion.fisher_information)
