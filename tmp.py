import numpy as np

from tropea_clustering import onion_multi

NDIM = 3


def main():
    ## Create the input data ###
    rng = np.random.default_rng(12345)
    input_data = np.array(
        [
            np.concatenate(
                (
                    rng.normal(0.0, 0.1, (500, NDIM)),
                    rng.normal(1.0, 0.1, (500, NDIM)),
                )
            )
            for _ in range(100)
        ]
    )

    delta_t = 10
    state_list, labels = onion_multi(input_data, delta_t)

    for state in state_list:
        print(state.mean, state.sigma, state.perc)

    _ = state_list[0].get_attributes()


if __name__ == "__main__":
    main()
