import numpy as np

from tropea_clustering import OnionUni, onion_uni


def main():
    ## Create the input data ###
    rng = np.random.default_rng(12345)
    input_data = [
        np.concatenate((rng.normal(0.0, 0.1, 500), rng.normal(1.0, 0.1, 500)))
        for _ in range(100)
    ]

    delta_t = 10

    # Test the class methods
    tmp = OnionUni(delta_t)
    tmp_params = {"bins": "auto", "number_of_sigmas": 2.0}
    tmp.set_params(**tmp_params)
    _ = tmp.get_params()

    state_list, labels = onion_uni(input_data, delta_t)

    print()
    for state in state_list:
        print(state.mean, state.sigma, state.perc)

    print(np.unique(labels))

    _ = state_list[0].get_attributes()


if __name__ == "__main__":
    main()
